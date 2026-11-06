"""Exact skipping on the chip path (docs/LAYER_SKIP.md).

The idea (credit: qwendoom, github.com/utsav1033/qwendoom, model/stepper.c; no code is shared): on any one token most
layers, heads and MLP units of the compiled chip do nothing, and that can be PROVEN from the weights and the token, so
the dense computation can be left out without changing a single bit of what comes out.

`replica_layer` is a plain-torch re-statement of one chip layer for ONE token (T == 1) with the same ops, dtypes and
order as transformers' Qwen3_5DecoderLayer / Qwen3_5GatedDeltaNet.  It is the reference for the probe (which counts what
is dead) and the base of the skipping layer; `check_replica` proves it bit-equal to the HF layer.
"""
import torch
import torch.nn.functional as F


def _l2(x, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


@torch.no_grad()
def replica_layer(layer, h, conv_state, rec):
    """One chip layer on a single token.  h: (1, 1, H) bf16.  conv_state (1, C, K) and rec (1, nv, dk, dv) are the layer's
    cache tensors BEFORE the token; they are not modified.  Returns (h_out, new_conv_state, new_rec, info)."""
    la = layer.linear_attn
    nv, nk, dk, dv = la.num_v_heads, la.num_k_heads, la.head_k_dim, la.head_v_dim
    x = layer.input_layernorm(h)
    mixed = la.in_proj_qkv(x).transpose(1, 2)                                   # (1, C, 1)
    z = la.in_proj_z(x).reshape(1, 1, -1, dv)
    b = la.in_proj_b(x); a = la.in_proj_a(x)
    new = torch.cat([conv_state, mixed], dim=-1).to(la.conv1d.weight.dtype)
    out = F.conv1d(new, la.conv1d.weight, None, padding=0, groups=new.shape[1])[:, :, -1:]
    out = F.silu(out).to(mixed.dtype)
    q, k, v = torch.split(out.transpose(1, 2), [la.key_dim, la.key_dim, la.value_dim], dim=-1)
    q = q.reshape(1, 1, -1, dk); k = k.reshape(1, 1, -1, dk); v = v.reshape(1, 1, -1, dv)
    beta = b.sigmoid()
    g = -la.A_log.float().exp() * F.softplus(a.float() + la.dt_bias)
    if nv // nk > 1:
        q = q.repeat_interleave(nv // nk, dim=2); k = k.repeat_interleave(nv // nk, dim=2)
    info = {"q_live": (q != 0).any(-1)[0, 0], "k_live": (k != 0).any(-1)[0, 0]}
    q, k, v, beta, dec = [t.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format) for t in (q, k, v, beta, g)]
    q = _l2(q); k = _l2(k); q = q / (q.shape[-1] ** 0.5)
    S = rec.to(v) * dec[:, :, 0].exp()[..., None, None]
    kv_mem = (S * k[:, :, 0].unsqueeze(-1)).sum(dim=-2)
    delta = (v[:, :, 0] - kv_mem) * beta[:, :, 0].unsqueeze(-1)
    S = S + k[:, :, 0].unsqueeze(-1) * delta.unsqueeze(-2)
    core = (S * q[:, :, 0].unsqueeze(-1)).sum(dim=-2)                          # (1, nv, dv)
    core = core.reshape(1, 1, nv, dv).contiguous().to(h.dtype)
    core = la.norm(core.reshape(-1, dv), z.reshape(-1, dv)).reshape(1, 1, -1)
    h1 = h + la.out_proj(core)
    x2 = layer.post_attention_layernorm(h1)
    act = layer.mlp.act_fn(layer.mlp.gate_proj(x2)) * layer.mlp.up_proj(x2)
    info["mlp_live"] = (act != 0)[0, 0]
    h2 = h1 + layer.mlp.down_proj(act)
    info["h1_eq"] = torch.equal(h1, h); info["h2_eq"] = torch.equal(h2, h)
    return h2, new[:, :, -conv_state.shape[-1]:], S, info


class Probe:
    """Counts, on real traces, which parts of each chip layer are provably or actually dead on single-token steps, and
    checks `replica_layer` against the HF layer bit for bit.  Attach to a merged model, run any program, read `rows`."""

    def __init__(self, model, first, last=None):
        self.model, self.first, self.rows, self.cur, self.replica_ok, self.replica_bad = model, first, [], None, 0, 0
        self._pending, self._hooks = {}, []
        for L in range(first, last if last is not None else model.config.num_hidden_layers):
            lyr = model.model.layers[L]
            self._hooks.append(lyr.register_forward_pre_hook(self._pre(L), with_kwargs=True))
            self._hooks.append(lyr.register_forward_hook(self._post(L), with_kwargs=True))

    def close(self):
        for h in self._hooks: h.remove()

    def _pre(self, L):
        def fn(mod, args, kwargs):
            h = args[0]; past = kwargs.get("past_key_values")
            if h.shape[1] != 1 or past is None or not past.has_previous_state(L, state_idx=0): self._pending[L] = None; return
            cl = past.layers[L]
            self._pending[L] = (h.clone(), cl.conv_states[0].clone(), cl.recurrent_states[0].clone())
        return fn

    def _post(self, L):
        def fn(mod, args, kwargs, out):
            p = self._pending.pop(L, None)
            if p is None: return
            h2, _, _, info = replica_layer(mod, *p)
            same = torch.equal(h2, out); self.replica_ok += same; self.replica_bad += (not same)
            q, k, m = info["q_live"], info["k_live"], info["mlp_live"]
            self.rows.append(dict(tok=self.cur, layer=L - self.first, q=int(q.sum()), k=int(k.sum()), heads=int((q | k).sum()),
                                  mlp=int(m.sum()), attn_dormant=not bool((q | k).any()), mlp_dormant=not bool(m.any()),
                                  passthrough=bool(torch.equal(out, p[0]))))
        return fn


def static_stats(model, first, last=None):
    """Per chip layer: heads and MLP units whose weights are not all zero (dead for every token), and the number of conv
    channels that read history (a tap at lag >= 1)."""
    res = []
    for L in range(first, last if last is not None else model.config.num_hidden_layers):
        lyr = model.model.layers[L]; la = lyr.linear_attn; nv, dk, dv, nk = la.num_v_heads, la.head_k_dim, la.head_v_dim, la.num_k_heads
        w = la.in_proj_qkv.weight
        qk_live = ((w[: nk * dk].reshape(nk, dk, -1) != 0).any(-1).any(-1) | (w[nk * dk: 2 * nk * dk].reshape(nk, dk, -1) != 0).any(-1).any(-1))
        out_live = (la.out_proj.weight.reshape(-1, nv, dv) != 0).any(0).any(-1)
        units = ((lyr.mlp.gate_proj.weight != 0).any(-1) | (lyr.mlp.up_proj.weight != 0).any(-1)) & (lyr.mlp.down_proj.weight != 0).any(0)
        cw = la.conv1d.weight[:, 0, :-1]
        res.append(dict(layer=L - first, heads=int((qk_live | out_live).sum()), of=nv, units=int(units.sum()), inter=units.numel(),
                        hist_channels=int((cw != 0).any(-1).sum()), conv_channels=cw.shape[0]))
    return res


# ----------------------------------------------------------------------------------------------------------------------
# Exact skipping implementation (docs/LAYER_SKIP.md)
# ----------------------------------------------------------------------------------------------------------------------
from qwentpu.driver import ChipPath, GraphedChip, TpuSession


class CompactLayer:
    """The structurally live part of one chip layer (weights only; true for every token), as dense compact matrices.

    heads  the v-heads whose q or k rows in in_proj_qkv are not all zero (the others never read or write: S1 in the doc)
    units  the MLP units with a non-zero gate row, up row and down column
    dense  True when the layer has conv taps at lag >= 1 (history channels): those layers keep the stock HF path
    """

    def __init__(self, layer):
        la, mlp = layer.linear_attn, layer.mlp
        nv, nk, dk, dv = la.num_v_heads, la.num_k_heads, la.head_k_dim, la.head_v_dim
        H = la.in_proj_qkv.weight.shape[1]; self.hidden = H; self.dk, self.dv = dk, dv
        cw = la.conv1d.weight[:, 0, :]
        self.dense = bool((cw[:, :-1] != 0).any()) or nv != nk
        if self.dense: return
        W = la.in_proj_qkv.weight; kd = nk * dk
        wq, wk, wv = W[:kd].reshape(nk, dk, H), W[kd: 2 * kd].reshape(nk, dk, H), W[2 * kd:].reshape(nv, dv, H)
        live = (wq != 0).any(-1).any(-1) | (wk != 0).any(-1).any(-1)
        hs = live.nonzero().flatten(); self.hs = hs; self.nh = int(hs.numel())
        if self.nh:
            w0 = cw[:, -1]
            self.Wq, self.Wk, self.Wv = (t[hs].reshape(-1, H).contiguous() for t in (wq, wk, wv))
            self.w0q = w0[:kd].reshape(nk, dk)[hs].reshape(-1).contiguous(); self.w0k = w0[kd: 2 * kd].reshape(nk, dk)[hs].reshape(-1).contiguous()
            self.w0v = w0[2 * kd:].reshape(nv, dv)[hs].reshape(-1).contiguous()
            self.Wz = la.in_proj_z.weight.reshape(nv, dv, H)[hs].reshape(-1, H).contiguous()
            self.Wa, self.Wb = la.in_proj_a.weight[hs].contiguous(), la.in_proj_b.weight[hs].contiguous()
            self.A, self.dt = la.A_log[hs].contiguous(), la.dt_bias[hs].contiguous()
            self.Wo = la.out_proj.weight.reshape(-1, nv, dv)[:, hs].reshape(la.out_proj.weight.shape[0], -1).contiguous()
        u = ((mlp.gate_proj.weight != 0).any(-1) & (mlp.up_proj.weight != 0).any(-1) & (mlp.down_proj.weight != 0).any(0)).nonzero().flatten()
        self.nu = int(u.numel())
        if self.nu:
            self.Wg, self.Wu = mlp.gate_proj.weight[u].contiguous(), mlp.up_proj.weight[u].contiguous()
            self.Wd = mlp.down_proj.weight[:, u].contiguous()


class SkipChip(ChipPath):
    """ChipPath that leaves out provably dead work (docs/LAYER_SKIP.md).  Single-token steps with a cache take the compact
    path (structurally dead heads / units dropped, a layer's attention half skipped when this token's q, k and decay prove it
    cannot read, write or decay anything); chunks and the first step use the stock layers.  The emitted tokens, the layer
    outputs and the recurrent states equal the stock path's bit for bit (tests/test_skip.py, results/a100_skip_*.log).

    The final readout (readout="lm_head") multiplies only the columns of the chip head that can be non-zero."""

    def __init__(self, model, v0, code_dims=None, readout="code", first=None, last=None):
        super().__init__(model, v0, code_dims, readout)
        if first is not None: self.first = first
        self.last = last if last is not None else model.config.num_hidden_layers
        self.cl = {L: CompactLayer(model.model.layers[L]) for L in range(self.first, self.last)}
        gain = 1.0 + model.model.norm.weight.float()
        self.cols = ((self.head != 0).any(0) & (gain != 0)).nonzero().flatten()
        self.head_c = self.head[:, self.cols].contiguous()
        self.stats = dict(attn_skipped=0, attn_run=0, steps=0)

    @torch.no_grad()
    def _flags(self, c, x):
        """q, k (post conv + silu), the per-head decay exponent g, and a (3,) bool: [some q != 0, some k != 0, some exp(g) != 1]."""
        qh = F.silu(F.linear(x, c.Wq) * c.w0q); kh = F.silu(F.linear(x, c.Wk) * c.w0k)
        g = -c.A.float().exp() * F.softplus(F.linear(x, c.Wa).float() + c.dt)
        return qh, kh, g, torch.stack([(qh != 0).any(), (kh != 0).any(), (g.exp() != 1).any()])

    @torch.no_grad()
    def _apply(self, c, layer, h, x, rec, qh, kh, g):
        """The attention half for the structurally live heads (the dense layer's arithmetic on fewer rows).  Returns h1."""
        n, dk, dv = c.nh, c.dk, c.dv
        vh = F.silu(F.linear(x, c.Wv) * c.w0v); z = F.linear(x, c.Wz).reshape(1, 1, n, dv)
        beta = F.linear(x, c.Wb).sigmoid()
        q = qh.reshape(1, 1, n, dk); k = kh.reshape(1, 1, n, dk); v = vh.reshape(1, 1, n, dv)
        q, k, v, beta, dec = [t.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format) for t in (q, k, v, beta, g)]
        q = _l2(q); k = _l2(k); q = q / (dk ** 0.5)
        S = rec[:, c.hs].to(v) * dec[:, :, 0].exp()[..., None, None]
        kv_mem = (S * k[:, :, 0].unsqueeze(-1)).sum(dim=-2)
        S = S + k[:, :, 0].unsqueeze(-1) * ((v[:, :, 0] - kv_mem) * beta[:, :, 0].unsqueeze(-1)).unsqueeze(-2)
        core = (S * q[:, :, 0].unsqueeze(-1)).sum(dim=-2)
        rec.index_copy_(1, c.hs, S.to(rec.dtype))
        core = core.reshape(1, 1, n, dv).contiguous().to(h.dtype)
        core = layer.linear_attn.norm(core.reshape(-1, dv), z.reshape(-1, dv)).reshape(1, 1, -1)
        return h + F.linear(core, c.Wo)

    @torch.no_grad()
    def _mlp(self, c, layer, h1):
        x2 = layer.post_attention_layernorm(h1)
        act = layer.mlp.act_fn(F.linear(x2, c.Wg)) * F.linear(x2, c.Wu)
        return h1 + F.linear(act, c.Wd)

    @torch.no_grad()
    def _attn(self, c, layer, h, rec):
        """Eager: returns h1; ONE host sync decides whether the attention half does anything on this token."""
        x = layer.input_layernorm(h)
        qh, kh, g, flags = self._flags(c, x)
        fq, fk, fd = flags.tolist()
        for key, f in (("q_live", fq), ("k_live", fk), ("decay", fd)): self.stats[key] = self.stats.get(key, 0) + int(f)
        if not (fq or fk or fd):
            self.stats["attn_skipped"] += 1; return h
        self.stats["attn_run"] += 1
        return self._apply(c, layer, h, x, rec, qh, kh, g)

    @torch.no_grad()
    def _step(self, L, h, past):
        layer = self.m.model.layers[L]; c = self.cl[L]
        if c.dense: return layer(h, position_embeddings=None, attention_mask=None, position_ids=None, past_key_values=past)
        if c.nh: h = self._attn(c, layer, h, past.layers[L].recurrent_states[0])
        if c.nu: h = self._mlp(c, layer, h)
        return h

    @torch.no_grad()
    def hidden(self, ids, past):
        m = self.m.model
        if ids.shape[1] != 1 or past is None or not all(past.has_previous_state(L, state_idx=0) for L in range(self.first, self.last)):
            h = m.embed_tokens(ids)
            for layer in m.layers[self.first: self.last]:
                h = layer(h, position_embeddings=None, attention_mask=None, position_ids=None, past_key_values=past)
            return m.norm(h)
        h = m.embed_tokens(ids); self.stats["steps"] += 1
        for L in range(self.first, self.last): h = self._step(L, h, past)
        return m.norm(h)

    @torch.no_grad()
    def readout_of(self, h):
        if self.readout == "code":
            bits = (h[..., self.code_dims] > 0).to(torch.int64)
            return (bits * (1 << torch.arange(bits.shape[-1], device=bits.device))).sum(-1)
        return (h[..., self.cols] @ self.head_c.T).argmax(-1)

    @torch.no_grad()
    def tokens(self, ids, past):
        return self.readout_of(self.hidden(ids, past))

    __call__ = tokens


class SkipGraphedChip(SkipChip, GraphedChip):
    """SkipChip with CUDA graphs: chunks (prompts) replay the stock chunk graphs; a single-token step replays, per chip layer, a
    test graph (the q / k / decay flags), the attention graph when the flags say it matters, and the MLP graph.  The host reads
    one 3-bool flag vector per layer that has live heads (one sync each) and nothing else; every layer graph is the same
    arithmetic as the eager SkipChip step, so the bits are the same."""

    def __init__(self, model, v0, code_dims=None, readout="code", first=None, last=None):
        super().__init__(model, v0, code_dims, readout, first, last)
        self.built = False; self.fallback = any(c.dense for c in self.cl.values())

    def attach(self, past):
        GraphedChip.attach(self, past); self.built = False

    @staticmethod
    def _capture(fn, restore):
        saved = [t.clone() for t in restore]
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2): fn()                                   # warm-up (allocator, autotune) off the graph
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): fn()
        for t, v in zip(restore, saved): t.copy_(v)                   # the warm-up runs mutated the live state: put it back
        return g

    @torch.no_grad()
    def _build(self):
        past, dev, m = self.past, self.m.device, self.m.model
        self.ids = torch.full((1, 1), self.v0, dtype=torch.long, device=dev)
        self.hb = torch.zeros(1, 1, self.m.config.hidden_size, dtype=torch.bfloat16, device=dev)
        self.out = torch.zeros(1, 1, dtype=torch.long, device=dev)
        recs = [past.layers[L].recurrent_states[0] for L in range(self.first, self.last)]
        restore = [self.hb] + recs
        self.g_embed = self._capture(lambda: self.hb.copy_(m.embed_tokens(self.ids)), restore)
        self.lg = {}
        for L in range(self.first, self.last):
            c, layer, rec = self.cl[L], m.layers[L], past.layers[L].recurrent_states[0]
            gt = gl = gm = None; fl = torch.zeros(3, dtype=torch.bool, device=dev)
            if c.nh:
                def test(c=c, layer=layer, fl=fl): fl.copy_(self._flags(c, layer.input_layernorm(self.hb))[3])
                def live(c=c, layer=layer, rec=rec):
                    x = layer.input_layernorm(self.hb); qh, kh, g, _ = self._flags(c, x)
                    self.hb.copy_(self._apply(c, layer, self.hb, x, rec, qh, kh, g))
                gt, gl = self._capture(test, restore), self._capture(live, restore)
            if c.nu:
                def mlp(c=c, layer=layer): self.hb.copy_(self._mlp(c, layer, self.hb))
                gm = self._capture(mlp, restore)
            self.lg[L] = (gt, gl, gm, fl)
        self.g_final = self._capture(lambda: self.out.copy_(self.readout_of(m.norm(self.hb))), restore)
        self.built = True

    @torch.no_grad()
    def tokens(self, ids, past):
        if ids.shape[1] != 1 or self.fallback: return GraphedChip.tokens(self, ids, past)
        assert past is self.past, "SkipGraphedChip is bound to the cache it was attached to"
        if not self.built: self._build()
        self.ids.copy_(ids); self.g_embed.replay(); self.stats["steps"] += 1
        for L in range(self.first, self.last):
            gt, gl, gm, fl = self.lg[L]
            if gt is not None:
                gt.replay()
                if any(fl.tolist()): gl.replay(); self.stats["attn_run"] += 1                # the one host sync of this layer
                else: self.stats["attn_skipped"] += 1
            if gm is not None: gm.replay()
        self.g_final.replay()
        return self.out.clone()

    __call__ = tokens


class SkipSession(TpuSession):
    """TpuSession whose chip path is SkipChip (graph=True: SkipGraphedChip).  Same feed() API: the host loop is unchanged."""

    def __init__(self, model, v0, readout="code", code_dims=None, graph=False, path=None):
        """path: a prebuilt eager SkipChip to reuse across sessions (its compact weights are built once)."""
        super().__init__(model, v0, fast=True, readout=readout, code_dims=code_dims)
        self.path = path if path is not None else (SkipGraphedChip if graph else SkipChip)(model, v0, code_dims, readout)
