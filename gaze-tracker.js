(function(){
  'use strict';

  // ── State ──
  const S = { IDLE:0, LOADING:1, CAMERA:2, MODEL:3, READY:4, ERROR:-1, NO_CAM:-2 };
  let state = S.IDLE, model = null, video = null, stream = null, animId = null;
  let gazeX = null, gazeY = null, active = false;
  let sX = null, sY = null, pvX = null, pvY = null, noFaceFrames = 0, fDetected = false;
  let calibSamples=[],calibPtIdx=-1,calibFrames=[],calibActive=false,calibOverlay=null;
  let statusText='',initialized=false,initStarted=false;
  const W=()=>window.innerWidth, H=()=>window.innerHeight;

  const PTS=[
    [0.1,0.1],[0.5,0.1],[0.9,0.1],
    [0.1,0.5],[0.5,0.5],[0.9,0.5],
    [0.1,0.9],[0.5,0.9],[0.9,0.9]
  ];
  const FR = 60, KEY = 'pl_gaze_calib', CAM_W=320, CAM_H=240;
  const SA=0.08, FA=0.35, TH=40;

  // ── Status bar (disabled) ──
  function showStatus(t){}

  // ── Dynamic script loader ──
  function loadScript(src){return new Promise((ok,no)=>{const s=document.createElement('script');s.src=src;s.onload=ok;s.onerror=()=>no(new Error('Failed to load '+src));document.head.appendChild(s)})}

  // ── TensorFlow + FaceMesh ──
  async function loadDeps(){
    if(typeof tf === 'undefined')
      try{await loadScript('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.21.0/dist/tf.min.js')}catch(e){showStatus('TF load fail');throw e}
    if(typeof faceLandmarksDetection === 'undefined')
      try{await loadScript('https://cdn.jsdelivr.net/npm/@tensorflow-models/face-landmarks-detection@1.0.6/dist/face-landmarks-detection.min.js')}catch(e){showStatus('FaceMesh load fail');throw e}
  }

  // ── Camera ──
  async function openCamera(){
    if(!navigator.mediaDevices||!navigator.mediaDevices.getUserMedia)throw new Error('no getUserMedia');
    stream = await navigator.mediaDevices.getUserMedia({
      audio:false,video:{facingMode:'user',width:{ideal:CAM_W},height:{ideal:CAM_H}}
    });
    video=document.createElement('video');
    video.srcObject=stream;video.playsInline=true;video.muted=true;
    video.setAttribute('playsinline','');
    video.style.cssText='position:fixed;top:-9999px;left:-9999px;opacity:0;pointer-events:none';
    document.body.appendChild(video);
    await video.play();
  }

  function closeCamera(){
    if(stream){stream.getTracks().forEach(t=>t.stop());stream=null}
    if(video&&video.parentNode){video.parentNode.removeChild(video);video=null}
  }

  // ── Head pose estimation ──
  function estimateHeadPose(kp){
    const eL=kp[33],eR=kp[263],n=kp[1],mL=kp[61],mR=kp[291];
    const fW=Math.hypot(eR.x-eL.x,eR.y-eL.y);
    if(fW<1)return null;
    const cx=(eL.x+eR.x)/2,cy=(eL.y+eR.y)/2, mx=(mL.x+mR.x)/2,my=(mL.y+mR.y)/2;
    const nox=(n.x-cx)/fW, noy=(n.y-cy)/fW;
    return{
      yaw:Math.max(-40,Math.min(40,nox*60)),
      pitch:Math.max(-30,Math.min(30,(noy-0.3)*50)),
      fs:fW
    };
  }

  // ── Iris + gaze estimation ──
  function extractGaze(kp,pose){
    const lI=kp[468],rI=kp[473];
    const lL=kp[33],lR=kp[133],rL=kp[362],rR=kp[263];
    const lT=kp[159],lB=kp[145],rT=kp[386],rB=kp[374];
    const lW=Math.hypot(lR.x-lL.x,lR.y-lL.y);
    const lH=Math.hypot(lB.x-lT.x,lB.y-lT.y);
    const rW=Math.hypot(rR.x-rL.x,rR.y-rL.y);
    const rH=Math.hypot(rB.x-rT.x,rB.y-rT.y);
    if(lW<1||rW<1)return null;
    const lCx=(lL.x+lR.x)/2,lCy=(lT.y+lB.y)/2;
    const rCx=(rL.x+rR.x)/2,rCy=(rT.y+rB.y)/2;
    const ix=((lI.x-lCx)/(lW*0.5)+(rI.x-rCx)/(rW*0.5))*0.5;
    const iy=((lI.y-lCy)/(lH*0.5)+(rI.y-rCy)/(rH*0.5))*0.5;
    return{ix,iy,yaw:pose.yaw,pitch:pose.pitch,fs:pose.fs};
  }

  // ── Uncalibrated projection ──
  function projRaw(ix,iy,y,p,f){
    const ds=Math.max(50,Math.min(300,f||80));
    const sc=W()/(ds*0.02), sr=H()/(ds*0.02);
    return{
      x:W()/2+ix*sc*1.2+y*6,
      y:H()/2+iy*sr*1.5-p*4
    };
  }

  // ── Calibration (Inverse Distance Weighting) ──
  function projCalib(ix,iy,y,p){
    if(!calibSamples||calibSamples.length<3)return null;
    let ws=0,wx=0,wy=0,eps=1e-6;
    for(const s of calibSamples){
      const d=Math.hypot(s.ix-ix,s.iy-iy,s.yaw-y,s.pitch-p);
      const w=1/(d*d+eps);
      ws+=w;wx+=w*s.sx;wy+=w*s.sy;
    }
    if(ws<eps)return null;
    return{x:wx/ws,y:wy/ws};
  }

  // ── Main detection loop ──
  async function detectLoop(){
    if(!initialized)return;
    if(!model||!video||video.readyState<2){animId=requestAnimationFrame(detectLoop);return}

    let faces;
    try{faces=await model.estimateFaces({input:video})}catch(e){faces=[]}

    if(faces&&faces.length>0){
      const kp=faces[0].keypoints;
      const pose=estimateHeadPose(kp);
      if(pose){
        const g=extractGaze(kp,pose);
        if(g){
          fDetected=true;noFaceFrames=0;
          let p=null;
          if(calibSamples&&calibSamples.length>=3)p=projCalib(g.ix,g.iy,g.yaw,g.pitch);
          if(!p)p=projRaw(g.ix,g.iy,g.yaw,g.pitch,g.fs);
          if(p){
            const spd=Math.hypot(p.x-(pvX||p.x),p.y-(pvY||p.y));
            const a=spd>TH?FA:SA;
            if(sX===null){sX=p.x;sY=p.y}
            else{sX+=(p.x-sX)*a;sY+=(p.y-sY)*a}
            pvX=p.x;pvY=p.y;
            gazeX=sX;gazeY=sY;
            active=true;
          }
          // calibration data collection
          if(calibActive&&calibPtIdx>=0){
            calibFrames.push(g);
            if(calibFrames.length>=FR){
              let six=0,siy=0,sy=0,sp=0,sf=0;
              for(const c of calibFrames){six+=c.ix;siy+=c.iy;sy+=c.yaw;sp+=c.pitch;sf+=c.fs}
              const n=calibFrames.length;
              calibSamples.push({ix:six/n,iy:siy/n,yaw:sy/n,pitch:sp/n,fs:sf/n,sx:PTS[calibPtIdx][0]*W(),sy:PTS[calibPtIdx][1]*H()});
              calibFrames=[];calibPtIdx++;
              updateCalibUI();
              if(calibPtIdx>=PTS.length){
                finishCalibration();
              }
            }
          }
        }
      }
    }else{
      noFaceFrames++;
      if(noFaceFrames>30)active=false;
    }

    if(initialized)animId=requestAnimationFrame(detectLoop);
  }

  // ── Calibration UI ──
  function buildCalibUI(){
    if(calibOverlay)calibOverlay.parentNode&&calibOverlay.parentNode.removeChild(calibOverlay);
    calibOverlay=document.createElement('div');
    calibOverlay.id='gazeCalibOverlay';
    calibOverlay.style.cssText='position:fixed;top:0;left:0;width:100%;height:100%;z-index:99998;background:rgba(0,0,0,0.85);display:flex;flex-direction:column;align-items:center;justify-content:center';
    calibOverlay.innerHTML=`<div style="text-align:center;color:rgba(255,255,255,0.5);font-family:'Inter',sans-serif;margin-bottom:20px">
      <div style="font-size:14px;letter-spacing:2px;text-transform:uppercase;color:rgba(255,240,180,0.6)">Калибровка взгляда</div>
      <div style="font-size:11px;margin-top:6px;color:rgba(255,255,255,0.3)">Смотрите на каждую точку, не двигая головой</div>
    </div>`;
    const dotContainer=document.createElement('div');
    dotContainer.style.cssText='position:relative;width:100%;height:60vh;max-width:600px';
    for(let i=0;i<PTS.length;i++){
      const d=document.createElement('div');
      d.id='gazeDot'+i;
      d.style.cssText='position:absolute;width:20px;height:20px;border-radius:50%;background:rgba(255,240,180,0.15);border:1px solid rgba(255,240,180,0.2);transform:translate(-50%,-50%);transition:all .3s;left:'+(PTS[i][0]*100)+'%;top:'+(PTS[i][1]*100)+'%';
      d.innerHTML='<div style="position:absolute;top:50%;left:50%;width:6px;height:6px;border-radius:50%;background:rgba(255,240,180,0.4);transform:translate(-50%,-50%);transition:all .3s"></div>';
      dotContainer.appendChild(d);
    }
    calibOverlay.appendChild(dotContainer);
    const btn=document.createElement('button');
    btn.id='gazeCalibCancel';
    btn.textContent='Отмена';
    btn.style.cssText='margin-top:20px;background:none;border:1px solid rgba(255,255,255,0.1);color:rgba(255,255,255,0.3);padding:8px 20px;font:11px Inter,sans-serif;letter-spacing:1px;text-transform:uppercase;cursor:pointer';
    btn.onclick=cancelCalibration;
    calibOverlay.appendChild(btn);
    document.body.appendChild(calibOverlay);
  }

  function updateCalibUI(){
    for(let i=0;i<PTS.length;i++){
      const d=document.getElementById('gazeDot'+i);
      if(!d)continue;
      const inner=d.querySelector('div');
      if(i===calibPtIdx){
        d.style.background='rgba(255,240,180,0.3)';d.style.borderColor='rgba(255,240,180,0.7)';
        d.style.transform='translate(-50%,-50%) scale(1.3)';
        if(inner){inner.style.background='rgba(255,240,180,0.9)';inner.style.width='10px';inner.style.height='10px'}
      }else if(i<calibPtIdx){
        d.style.background='rgba(60,200,100,0.2)';d.style.borderColor='rgba(60,200,100,0.4)';
        d.style.transform='translate(-50%,-50%) scale(1)';
        if(inner){inner.style.background='rgba(60,200,100,0.6)';inner.style.width='6px';inner.style.height='6px'}
      }else{
        d.style.background='rgba(255,240,180,0.15)';d.style.borderColor='rgba(255,240,180,0.2)';
        d.style.transform='translate(-50%,-50%) scale(1)';
        if(inner){inner.style.background='rgba(255,240,180,0.4)';inner.style.width='6px';inner.style.height='6px'}
      }
    }
    if(calibPtIdx>=0&&calibPtIdx<PTS.length){
      const pct=Math.round((calibFrames.length/FR)*100);
      showStatus('Калибровка: точка '+(calibPtIdx+1)+'/'+PTS.length+' ('+pct+'%)');
    }
  }

  function finishCalibration(){
    calibActive=false;calibPtIdx=-1;
    if(calibOverlay&&calibOverlay.parentNode)calibOverlay.parentNode.removeChild(calibOverlay);
    calibOverlay=null;
    try{localStorage.setItem(KEY,JSON.stringify(calibSamples))}catch(e){}
    showStatus('Калибровка завершена');
    setTimeout(()=>showStatus(''),3000);
  }

  function cancelCalibration(){
    calibActive=false;calibPtIdx=-1;calibFrames=[];calibSamples=[];
    if(calibOverlay&&calibOverlay.parentNode)calibOverlay.parentNode.removeChild(calibOverlay);
    calibOverlay=null;
    showStatus('');
  }

  // ── Load saved calibration ──
  function loadCalib(){
    try{
      const d=localStorage.getItem(KEY);
      if(d){calibSamples=JSON.parse(d);return calibSamples.length>=3}
    }catch(e){}
    return false;
  }

  // ── Public API ──
  const GazeTracker = {
    async start(){
      if(initialized)return;
      if(initStarted)return;
      if('ontouchstart' in window && !/Mac|Win/i.test(navigator.platform))return;
      initStarted=true;
      state=S.LOADING;showStatus('Загрузка модели...');
      try{
        await loadDeps();
        showStatus('Открытие камеры...');
        state=S.CAMERA;
        await openCamera();
        showStatus('Загрузка FaceMesh...');
        state=S.MODEL;
        model = await faceLandmarksDetection.load(
          faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh,
          {maxFaces:1,refineLandmarks:true}
        );
        loadCalib();
        state=S.READY;initialized=true;
        showStatus('Eye tracking активен');
        setTimeout(()=>showStatus(''),2000);
        detectLoop();



      }catch(e){
        state=S.ERROR;
        showStatus('Eye tracking недоступен: '+(e.message||'ошибка'));
        setTimeout(()=>showStatus(''),5000);
        closeCamera();
        initStarted=false;
      }
    },
    stop(){
      if(animId){cancelAnimationFrame(animId);animId=null}
      if(calibActive){
        calibActive=false;calibPtIdx=-1;calibFrames=[];
        if(calibOverlay&&calibOverlay.parentNode)calibOverlay.parentNode.removeChild(calibOverlay);
        calibOverlay=null;
      }
      closeCamera();
      active=false;initialized=false;initStarted=false;
      gazeX=null;gazeY=null;sX=null;sY=null;
      state=S.IDLE;

    },
    isActive(){return active&&initialized&&state===S.READY},
    getGaze(){return active?{x:gazeX,y:gazeY}:null},
    async calibrate(){
      if(calibActive)return;
      if(!initialized){await this.start()}
      if(calibActive||!initialized)return;
      calibSamples=[];calibFrames=[];calibPtIdx=0;calibActive=true;
      buildCalibUI();updateCalibUI();
      showStatus('Калибровка: точка 1/'+PTS.length);
    },
    isCalibrated(){return calibSamples&&calibSamples.length>=3},
    getStatus(){return state}
  };

  window.GazeTracker = GazeTracker;
})();
