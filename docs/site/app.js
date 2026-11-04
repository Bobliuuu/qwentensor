(function(){
  var R = window.RESULTS || {headline:{},columns:[],rows:[],ops:[]};
  function el(t,c,x){var e=document.createElement(t);if(c)e.className=c;if(x!=null)e.textContent=x;return e}

  // headline stats
  var H=R.headline||{}, labels={networks_run:"networks bit-exact on the real model",tokens_replayed:"token positions replayed",mismatches:"mismatches vs reference",qwen_models:"Qwen3.5 sizes bit-exact on a GPU"};
  var st=document.getElementById("stats");
  Object.keys(labels).forEach(function(k){
    var v=H[k], d=el("div","stat"), n=el("div","n"+(v==null?" pending":""),v==null?"pending":Number(v).toLocaleString());
    d.appendChild(n);d.appendChild(el("div","l",labels[k]));st.appendChild(d);
  });

  // pipeline stepper
  var steps=[
   ["PyTorch","your nn.Module","Write a torch module. qt.compile traces it with torch.fx and lowers it to the float IR. Unsupported ops raise at compile time, naming the op."],
   ["Quantize","calibration data","A calibration set fixes integer scales per tensor and a per-layer shift (post-training quantization, optionally STE fine-tuning). Activations are 0..31, weights -16..15."],
   ["Integer graph","IR","The quantized network becomes an integer-only graph (see docs/IR.md). A NumPy reference executes the same graph and defines the expected output."],
   ["Prompt","token program","The graph is lowered to a prompt of tokens: weight cells, program-memory words (successor, destination, column mask, shift) and the input. Qwen3.5 carries the qwenTPU layers compiled in."],
   ["Qwen emits","the result","The model generates one token per neuron; the host only does argmax and append. The tokens are decoded and compared with the reference. Verified on a CPU simulator of the circuit, and on Qwen3.5-0.8B (A100) for four small programs (see docs/EXECUTOR.md)."]];
  var pipe=document.getElementById("pipe"), det=document.getElementById("detail");
  function show(i){
    [].forEach.call(pipe.children,function(b,j){b.setAttribute("aria-pressed",j===i)});
    det.innerHTML="";det.appendChild(el("h3",null,(i+1)+". "+steps[i][0]));det.appendChild(el("p",null,steps[i][2]));
  }
  steps.forEach(function(s,i){var b=el("button");b.appendChild(document.createTextNode((i+1)+". "+s[0]));b.appendChild(el("small",null,s[1]));b.onclick=function(){show(i)};pipe.appendChild(b)});
  show(0);

  // ops table with filter
  var ops=R.ops||[], ot=document.getElementById("opstable"), cur="all";
  var names={yes:"supported",no:"unsupported",tbd:"to be confirmed"};
  function drawOps(){
    ot.innerHTML="";var h=el("tr");["Operation","Status","Note"].forEach(function(x){h.appendChild(el("th",null,x))});ot.appendChild(h);
    ops.filter(function(o){return cur==="all"||o.status===cur}).forEach(function(o){
      var r=el("tr");r.appendChild(el("td",null,o.op));var td=el("td");td.appendChild(el("span","pill "+o.status,names[o.status]||o.status));r.appendChild(td);r.appendChild(el("td",null,o.note||""));ot.appendChild(r)});
  }
  var f=document.getElementById("opfilter");
  ["all","yes","no","tbd"].forEach(function(k){var b=el("button",null,k==="all"?"all":names[k]);b.setAttribute("aria-pressed",k===cur);b.onclick=function(){cur=k;[].forEach.call(f.children,function(c){c.setAttribute("aria-pressed",c===b)});drawOps()};f.appendChild(b)});
  drawOps();

  // results table
  var rt=document.getElementById("results-table"), cols=R.columns||[], rows=R.rows||[];
  var hr=el("tr");cols.forEach(function(c){hr.appendChild(el("th",null,c.label))});rt.appendChild(hr);
  if(!rows.length){var tr=el("tr"),td=el("td","empty","No results recorded yet. This table reads docs/site/data/results.json.");td.colSpan=cols.length||1;tr.appendChild(td);rt.appendChild(tr)}
  rows.forEach(function(row){var tr=el("tr");cols.forEach(function(c){var v=row[c.key],td=el("td",v==null?"pending":null,v==null?"pending":String(v));tr.appendChild(td)});rt.appendChild(tr)});
})();
