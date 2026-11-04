// Shared helpers for the light, dashboard-style explainer clips (white bg, big type, plain-language captions).
const W=1920,H=1080,cvs=document.getElementById('c');cvs.width=W;cvs.height=H;const ctx=cvs.getContext('2d');
const clamp=(x,a=0,b=1)=>Math.max(a,Math.min(b,x)),prog=(t,a,b)=>clamp((t-a)/(b-a)),eo=x=>1-Math.pow(1-x,3),eio=x=>x<.5?4*x*x*x:1-Math.pow(-2*x+2,3)/2,lerp=(a,b,x)=>a+(b-a)*x;
const eob=x=>{const c1=1.70158,c3=c1+1;return 1+c3*Math.pow(x-1,3)+c1*Math.pow(x-1,2)};
const comma=n=>Math.round(n).toLocaleString('en-US');
const C={bg:'#ffffff',ink:'#141a26',mut:'#6b7488',line:'#dfe4ec',panel:'#f5f7fb',blue:'#2563eb',bluel:'#dbe7ff',or:'#f08a24',orl:'#fff0dd',gr:'#16a34a',grl:'#dcf5e4',gy:'#cbd1dc',gyl:'#e9edf3',pu:'#7c3aed',pul:'#efe6fb',pk:'#e0455f',pkl:'#ffe1e6'};
const SANS='"Helvetica Neue",Helvetica,Arial,sans-serif',MONO='Menlo,"SF Mono",Monaco,monospace';
function rr(x,y,w,h,r,fill,stroke,lw=2){ctx.beginPath();ctx.roundRect(x,y,w,h,r);if(fill){ctx.fillStyle=fill;ctx.fill()}if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=lw;ctx.stroke()}}
function tx(s,x,y,o={}){const{size:z=32,weight:wt=600,color:c=C.ink,align:a='left',mono:m=false,alpha:al=1,ls:l=0}=o;ctx.save();ctx.globalAlpha*=al;ctx.font=`${wt} ${z}px ${m?MONO:SANS}`;ctx.fillStyle=c;ctx.textAlign=a;ctx.letterSpacing=l+'px';ctx.fillText(s,x,y);ctx.restore()}
function ln(x1,y1,x2,y2,c,w=3,d){ctx.save();ctx.strokeStyle=c;ctx.lineWidth=w;if(d)ctx.setLineDash(d);ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();ctx.restore()}
function dotc(x,y,r,c,g){ctx.save();if(g){ctx.shadowColor=c;ctx.shadowBlur=g}ctx.fillStyle=c;ctx.beginPath();ctx.arc(x,y,r,0,7);ctx.fill();ctx.restore()}
function mixc(a,b,x){const p=h=>[parseInt(h.slice(1,3),16),parseInt(h.slice(3,5),16),parseInt(h.slice(5,7),16)];const A=p(a),B=p(b);return `rgb(${A.map((v,i)=>Math.round(lerp(v,B[i],x))).join(',')})`}
const sgn=v=>(v>0?'+':v<0?'−':'')+Math.abs(v);
function header(n,title,u){const a=eo(prog(u,0,.5));ctx.save();ctx.globalAlpha=a;rr(48,24,230,54,12,C.blue);tx('qwentensor',163,63,{size:32,weight:800,align:'center',color:'#fff'});tx(n+'  '+title,302,64,{size:38,weight:800});
 tx('Any PyTorch NN, run inside Qwen3.5',W-48,62,{size:28,weight:500,align:'right',color:C.mut});ln(48,96,W-48,96,C.line,2);ctx.restore()}
// bottom caption: plain sentence (big) + precise terms (smaller)
function caption(caps,u){for(const [a,b,plain,sub] of caps){if(u>=a&&u<=b){const k=Math.min(prog(u,a,a+.35),1-prog(u,b-.3,b));ctx.save();ctx.globalAlpha=k;
  rr(48,906,W-96,150,18,C.panel,C.line,2);tx(plain,W/2,972,{size:48,weight:800,align:'center'});tx(sub,W/2,1026,{size:30,weight:500,align:'center',color:C.mut});ctx.restore()}}}
function register(dur,fn){window.DUR=dur;window.render=u=>{ctx.setTransform(1,0,0,1,0,0);ctx.globalAlpha=1;ctx.fillStyle=C.bg;ctx.fillRect(0,0,W,H);fn(u);const e=1-prog(u,dur-.3,dur);if(e<1){ctx.save();ctx.globalAlpha=1-e;ctx.fillStyle='#fff';ctx.fillRect(0,0,W,H);ctx.restore()}};window.READY=true}
