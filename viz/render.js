#!/usr/bin/env node
// node render.js <name...> [--gif] [--stills t1,t2] [--fps 30] [--gifw 960] [--giffps 15] [--colors 96]   (pages: name.html in this folder)
// node render.js --svg diagrams/foo.svg ...   (rasterise static SVG/HTML to viz/images/<name>.png at 2x)
// Same pipeline as qwenTPU social/anim/render_v2.js: headless Chrome via puppeteer-core -> ffmpeg. puppeteer-core is borrowed from the qwenTPU checkout.
const fs=require('fs'),path=require('path'),{spawn,spawnSync}=require('child_process');
const PPT=process.env.PPT||path.join(process.env.HOME,'qwentpu_src/qwentpu/social/video/node_modules/puppeteer-core');
const puppeteer=require(PPT);
const args=process.argv.slice(2);const flag=k=>args.includes('--'+k);const val=k=>{const i=args.indexOf('--'+k);return i<0?null:args[i+1]};
const OUT=path.join(__dirname,'out');fs.mkdirSync(path.join(OUT,'stills'),{recursive:true});
const CHROME=process.env.CHROME||'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const FPS=Number(val('fps')||30);
const skip=new Set(['--stills','--fps','--gifw','--giffps','--colors']);
const names=args.filter((a,i)=>!a.startsWith('--')&&!skip.has(args[i-1]));
async function open(browser,name){const p=await browser.newPage();await p.setViewport({width:1920,height:1080,deviceScaleFactor:1});
 p.on('pageerror',e=>console.log('PAGEERR',name,e.message));p.on('console',m=>{if(m.type()==='error')console.log('CONSOLE',name,m.text())});
 await p.goto('file://'+path.join(__dirname,name+'.html'));await p.waitForFunction('window.READY===true',{timeout:90000});return p}
async function renderOne(browser,name){
 const p=await open(browser,name);const dur=await p.evaluate(()=>window.DUR);const n=Math.round(dur*FPS);
 const file=path.join(OUT,name+'.mp4');
 const ff=spawn('ffmpeg',['-y','-loglevel','error','-f','image2pipe','-framerate',String(FPS),'-c:v','mjpeg','-i','-','-c:v','libx264','-preset','medium','-crf','17','-pix_fmt','yuv420p','-r',String(FPS),'-movflags','+faststart',file],{stdio:['pipe','inherit','inherit']});
 const done=new Promise(r=>ff.on('close',r));
 for(let i=0;i<n;i++){await p.evaluate(t=>window.render(t),i/FPS);const b=await p.screenshot({type:'jpeg',quality:95});if(!ff.stdin.write(b))await new Promise(r=>ff.stdin.once('drain',r))}
 ff.stdin.end();await done;await p.close();console.log('wrote',file,dur.toFixed(1)+'s',(fs.statSync(file).size/1e6).toFixed(2)+' MB');
 if(flag('gif')){const g=path.join(OUT,name+'.gif'),w=val('gifw')||960,fp=val('giffps')||15,col=val('colors')||96;
  spawnSync('ffmpeg',['-y','-loglevel','error','-i',file,'-vf',`fps=${fp},scale=${w}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=${col}:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle`,'-loop','0',g]);console.log('wrote',g,(fs.statSync(g).size/1e6).toFixed(2)+' MB')}
}
(async()=>{
 const browser=await puppeteer.launch({executablePath:CHROME,headless:'new',args:['--hide-scrollbars','--force-device-scale-factor=1','--font-render-hinting=none']});
 if(flag('svg')){fs.mkdirSync(path.join(__dirname,'images'),{recursive:true});
  for(const f of names){const p=await browser.newPage();await p.setViewport({width:1600,height:900,deviceScaleFactor:2});await p.goto('file://'+path.resolve(__dirname,f));
   const el=await p.$('svg');await el.screenshot({path:path.join(__dirname,'images',path.basename(f).replace(/\.\w+$/,'')+'.png'),omitBackground:false});await p.close();console.log('png',f)}
  await browser.close();return}
 if(val('stills')){for(const name of names){const p=await open(browser,name);for(const t of val('stills').split(',').map(Number)){await p.evaluate(t=>window.render(t),t);await p.screenshot({path:path.join(OUT,'stills',`${name}_${t.toFixed(1)}.png`)})}await p.close()}await browser.close();return}
 let q=[...names];await Promise.all([0,1].map(async()=>{while(q.length)await renderOne(browser,q.shift())}));
 await browser.close();
})();
