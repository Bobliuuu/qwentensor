// Helpers for the qwentensor clips. Needs light.js and data/trace.js (window.TRACE, window.RESULTS).
// Everything drawn comes from TRACE/RESULTS; TRACE.placeholder=true prints an "illustrative data" tag on screen.
const D=window.TRACE,R=window.RESULTS;
const PH=!!D.placeholder,RPH=!!R.placeholder;
const SRC_REAL=D.source==='real-model';
function mixc(a,b,x){const p=h=>h[0]==='#'?[parseInt(h.slice(1,3),16),parseInt(h.slice(3,5),16),parseInt(h.slice(5,7),16)]:h.match(/[\d.]+/g).slice(0,3).map(Number);const A=p(a),B=p(b);return `rgb(${A.map((v,i)=>Math.round(v+(B[i]-v)*x)).join(',')})`}
const KIND={READ:{c:'#64748b',l:'#e6ebf3'},ACT:{c:C.blue,l:C.bluel},OUT:{c:C.gr,l:C.grl},HALT:{c:C.pu,l:C.pul},PROG:{c:C.pu,l:C.pul},CELL:{c:C.or,l:C.orl},WGT:{c:C.or,l:C.orl},INP:{c:C.gr,l:C.grl}};
function chip(s,x,y,kind,o={}){const k=KIND[kind]||KIND.READ,z=o.size||26;ctx.save();ctx.font=`800 ${z}px ${MONO}`;const w=ctx.measureText(s).width+(o.pad||28);ctx.restore();
 ctx.save();ctx.globalAlpha*=o.alpha??1;rr(x,y,w,z+22,(z+22)/2,o.fill||k.l,k.c,2.5);tx(s,x+w/2,y+z+4,{size:z,weight:800,align:'center',mono:true,color:k.c});ctx.restore();return w}
function chipC(s,cx,cy,kind,o={}){const z=o.size||28;ctx.save();ctx.font=`800 ${z}px ${MONO}`;const w=ctx.measureText(s).width+30;ctx.restore();return chip(s,cx-w/2,cy-(z+22)/2,kind,o)}
function tokLabel(t){return t.kind==='READ'?'READ op'+t.k:t.kind==='ACT'?'ACT r'+t.k+' = '+t.val:'HALT'}
function arrow(x1,y1,x2,y2,c,w=5,dash){ln(x1,y1,x2,y2,c,w,dash);const a=Math.atan2(y2-y1,x2-x1),L=20;ctx.save();ctx.fillStyle=c;ctx.beginPath();ctx.moveTo(x2,y2);ctx.lineTo(x2-L*Math.cos(a-.4),y2-L*Math.sin(a-.4));ctx.lineTo(x2-L*Math.cos(a+.4),y2-L*Math.sin(a+.4));ctx.fill();ctx.restore()}
function alongPath(pts,f){let L=0;const seg=[];for(let i=1;i<pts.length;i++){const d=Math.hypot(pts[i][0]-pts[i-1][0],pts[i][1]-pts[i-1][1]);seg.push(d);L+=d}let s=f*L;for(let i=0;i<seg.length;i++){if(s<=seg[i]||i===seg.length-1){const g=clamp(s/seg[i]);return[lerp(pts[i][0],pts[i+1][0],g),lerp(pts[i][1],pts[i+1][1],g)]}s-=seg[i]}return pts[pts.length-1]}
function box(x,y,w,h,title,sub,col,fill,hl=0,ts=46){rr(x,y,w,h,20,fill,col,3+hl*3);if(hl>0){ctx.save();ctx.globalAlpha=hl*.5;rr(x-6,y-6,w+12,h+12,24,null,col,3);ctx.restore()}
 tx(title,x+w/2,y+(sub?h/2-4:h/2+ts/3),{size:ts,weight:800,align:'center',color:col});if(sub)tx(sub,x+w/2,y+h/2+38,{size:28,weight:600,align:'center',color:C.mut})}
// code with light syntax colouring; n = number of characters revealed. Returns nothing.
const KW=/^(class|def|return|import|from|self|super)$/;
function codeSeg(line){const out=[];const re=/(\s+|[A-Za-z_][A-Za-z_0-9]*|\d+|.)/g;let m;while((m=re.exec(line))){const s=m[0];let c=C.ink;if(KW.test(s))c=C.pu;else if(/^\d+$/.test(s))c=C.or;else if(/^(nn|F|torch)$/.test(s))c=C.blue;else if(/^(Linear|Module|relu)$/.test(s))c=C.gr;out.push([s,c])}return out}
function drawCode(lines,n,x,y,lh,size){ctx.save();ctx.font=`600 ${size}px ${MONO}`;const cw=ctx.measureText('M').width;ctx.restore();let left=n,cur=null;
 lines.forEach((l,i)=>{if(left<=0)return;let px=x;for(const [s,c] of codeSeg(l)){if(left<=0)break;const t=s.slice(0,left);tx(t,px,y+i*lh,{size,weight:600,color:c,mono:true});px+=cw*t.length;left-=t.length}if(left<=0&&!cur)cur=[px,y+i*lh];left-=1});
 return cur}
// honesty tag: where the token trace was produced
function placeholderTag(u,x=W-48,y=130){const msg=PH?'illustrative data (synthetic, not from a Qwen run)':SRC_REAL?'trace verified on the real Qwen3.5-0.8B (A100)':'CPU-simulator trace of the real quantized net. Same machine verified exact on real Qwen3.5-0.8B (A100, digits.log 40/40)';
 tx(msg,x,y,{size:22,weight:700,align:'right',color:SRC_REAL?C.gr:C.pk})}
const accTxt=v=>v==null?'--':(v*100).toFixed(1)+'%';
// proportional token strip with a cap on drawn cells. kinds: array of 'CELL'|'PROG'|'ACT' per prompt token. fill 0..1 reveals left to right.
// Returns {per: tokens per drawn cell}. Draws at most maxCells cells over width w.
function tokenStrip(kinds,x,y,w,h,fill,maxCells=150){const n=kinds.length,per=Math.max(1,Math.ceil(n/maxCells)),m=Math.ceil(n/per),cw=w/m;
 const show=Math.floor(m*fill);for(let j=0;j<show;j++){const k=kinds[Math.min(n-1,j*per+(per>>1))],K=KIND[k];rr(x+j*cw+1,y,Math.max(2,cw-2),h,Math.min(5,cw/3),K.l,K.c,1.5)}return{per,cells:m}}
