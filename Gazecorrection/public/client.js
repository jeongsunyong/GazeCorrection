var socket = io.connect('http://localhost:8005/setting', {
    path: '/socket.io'
  });

var isChannelReady = false;
var isInitiator = false;
var isStarted = false;
var localStream;
var pc;
var remoteStream;
var turnReady;

var pcConfig = {
  'iceServers': [{
    'urls': 'stun:stun.l.google.com:19302'
  }]
};

//슬라이드
var slider_1 = document.getElementById("myRange_1");
var output_1 = document.getElementById("demo_1");
var slider_2 = document.getElementById("myRange_2");
var output_2 = document.getElementById("demo_2");

// Display the default slider value
output_1.innerHTML = slider_1.value;
output_2.innerHTML = slider_2.value;

// Update the current slider value (each time you drag the slider handle)
slider_1.oninput = function() {
  socket.emit('wgaze', this.value);
  output_1.innerHTML = this.value;
}
slider_2.oninput = function() {
  socket.emit('pupil', this.value);
  output_2.innerHTML = this.value;
}

//연동추가시작
var configuration = null;
var photo = document.getElementById('setphoto');
var canvas = document.getElementById('setcanvas');
var photoContext = photo.getContext('2d');

var trail = document.getElementById('trail');
var snapBtn = document.getElementById('snap');
var sendBtn = document.getElementById('send');
var snapAndSendBtn = document.getElementById('snapAndSend');

var photoContextW;
var photoContextH;

var resultimgContextW;
var resultimgContextH;

// function grabWebCamVideo() {
//   console.log('Getting user media (video) ...');
//   navigator.mediaDevices.getUserMedia({
//     audio: false,
//     video: true
//   })
//   .then(gotStream)
//   .catch(function(e) {
//     alert('getUserMedia() error: ' + e.name);
//   });
// }
//연동추가끝
// Set up audio and video regardless of what devices are present.
var sdpConstraints = {
  offerToReceiveAudio: true,
  offerToReceiveVideo: true
};

/////////////////////////////////////////////
var room;

function switchid(roomId){
  room = roomId;
  console.log('room id in client:' + room);
}

socket.on('roomid', function (roomId) {
  switchid(roomId);
});

socket.emit('create or join', room);

socket.on('created', function(room) {
  console.log('Created room ' + room);
  isInitiator = true;
});

socket.on('join', function (room){
  console.log('Another peer made a request to join room ' + room);
  console.log('This peer is the initiator of room ' + room + '!');
  isChannelReady = true;
});

socket.on('joined', function(room) {
  console.log('joined: ' + room);
  isChannelReady = true;
});

socket.on('log', function(array) {
  console.log.apply(console, array);
});

////////////////////////////////////////////////

function sendMessage(message) {
  console.log('Client sending message: ', message);
  socket.emit('message', message);
}

// This client receives a message
socket.on('message', function(message) {
  console.log('Client received message:', message);
  if (message === 'got user media') {
    maybeStart();
  } else if (message.type === 'offer') {
    if (!isInitiator && !isStarted) {
      maybeStart();
    }
    pc.setRemoteDescription(new RTCSessionDescription(message));
    doAnswer();
  } else if (message.type === 'answer' && isStarted) {
    pc.setRemoteDescription(new RTCSessionDescription(message));
  } else if (message.type === 'candidate' && isStarted) {
    var candidate = new RTCIceCandidate({
      sdpMLineIndex: message.label,
      candidate: message.candidate
    });
    pc.addIceCandidate(candidate);
  } else if (message === 'bye' && isStarted) {
    handleRemoteHangup();
  }
});

////////////////////////////////////////////////////
var localVideo = document.querySelector('#settingVideo');

var clientnumber = 0;

socket.on('GetClientNum',function(clientnum){
  clientnumber = clientnum;
  getDevices().then(gotDevices).then(getStream);
})

function getDevices() {
  if (localStream) {
    localStream.getTracks().forEach(track => {
    track.stop();
  });
}
  return navigator.mediaDevices.enumerateDevices();
}

var SourceId = "";

function gotDevices(deviceInfos) {
  var checkDevice= 0;
  for (const deviceInfo of deviceInfos) {
    if (deviceInfo.kind === 'videoinput') {
      if(clientnumber % 2 == 0 && checkDevice == 0)
      {
        SourceId = deviceInfo.deviceId;
        break;
      }
      else if(checkDevice % 2 == 1 && checkDevice == 1)
      {
        SourceId = deviceInfo.deviceId;
        break;
      }
      checkDevice++;
    }
  }
}

function handleError(error) {
  console.error('Error: ', error);
}

function getStream() {
  var constraints = {
    video: {deviceId: SourceId ? {exact: SourceId} : undefined}
  };

  return navigator.mediaDevices.getUserMedia(constraints).then(gotStream).catch(handleError);

}

function gotStream(stream) {
  console.log('Adding local stream.');
  localStream = stream;
  localVideo.srcObject = stream;
  sendMessage('got user media');
  if (isInitiator) {
    maybeStart();
  }
  show();//추가
}

if (location.hostname !== 'localhost') {
  requestTurn(
    'https://computeengineondemand.appspot.com/turn?username=41784574&key=4080218913'
  );
}

function maybeStart() {
  console.log('>>>>>>> maybeStart() ', isStarted, localStream, isChannelReady);
  if (!isStarted && typeof localStream !== 'undefined' && isChannelReady) {
    console.log('>>>>>> creating peer connection');
    createPeerConnection();
    pc.addStream(localStream);
    isStarted = true;
    console.log('isInitiator', isInitiator);
    if (isInitiator) {
      doCall();
    }
  }
}

window.onbeforeunload = function() {
  sendMessage('bye');
};

/////////////////////////////////////////////////////////

function createPeerConnection() {
  try {
    pc = new RTCPeerConnection(null);
    pc.onicecandidate = handleIceCandidate;
    pc.onaddstream = handleRemoteStreamAdded;
    pc.onremovestream = handleRemoteStreamRemoved;
    console.log('Created RTCPeerConnnection');
  } catch (e) {
    console.log('Failed to create PeerConnection, exception: ' + e.message);
    alert('Cannot create RTCPeerConnection object.');
    return;
  }
}

function handleIceCandidate(event) {
  console.log('icecandidate event: ', event);
  if (event.candidate) {
    sendMessage({
      type: 'candidate',
      label: event.candidate.sdpMLineIndex,
      id: event.candidate.sdpMid,
      candidate: event.candidate.candidate
    });
  } else {
    console.log('End of candidates.');
  }
}

function handleCreateOfferError(event) {
  console.log('createOffer() error: ', event);
}

function doCall() {
  console.log('Sending offer to peer');
  pc.createOffer(setLocalAndSendMessage, handleCreateOfferError);
}

function doAnswer() {
  console.log('Sending answer to peer.');
  pc.createAnswer().then(
    setLocalAndSendMessage,
    onCreateSessionDescriptionError
  );
}

function setLocalAndSendMessage(sessionDescription) {
  pc.setLocalDescription(sessionDescription);
  console.log('setLocalAndSendMessage sending message', sessionDescription);
  sendMessage(sessionDescription);
}

function onCreateSessionDescriptionError(error) {
  trace('Failed to create session description: ' + error.toString());
}

function requestTurn(turnURL) {
  var turnExists = true;
  if (!turnExists) {
  }
}

function handleRemoteStreamAdded(event) {
  console.log('Remote stream added.');
  remoteStream = event.stream;
  remoteVideo.srcObject = remoteStream;
}

function handleRemoteStreamRemoved(event) {
  console.log('Remote stream removed. Event: ', event);
}

function hangup() {
  console.log('Hanging up.');
  stop();
  sendMessage('bye');
}

function handleRemoteHangup() {
  console.log('Session terminated.');
  stop();
  isInitiator = false;
}

function stop() {
  isStarted = false;
  pc.close();
  pc = null;
}

var blob;
function snapPhoto() {
    console.log("snapPhoto")
    photoContext.drawImage(localVideo, 0, 0, photo.width, photo.height);
    //photo = photoContext.getImageData(0,0,480,640);
    photo = photoContext.getImageData(0,0,640,480);
    canvas.src = ($('#setphoto')[0]).toDataURL("image/png");

    var imageData = canvas.src;
    var block = imageData.split(";");
    var contentType = block[0].split(":")[1];

    var byteCharacters = atob(imageData.replace(/^data:image\/(png|jpeg|jpg);base64,/,''));

    var byteNumbers = new Array(byteCharacters.length);
    for (var i = 0; i < byteCharacters.length; i++) {
      byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    var byteArray = new Uint8Array(byteNumbers);

    blob = new Blob([byteArray], {type: contentType});
    socket.emit('send',blob, clientnumber);
}

setInterval(snapPhoto,200);

socket.on('get',function(res){
  var base64d = res.toString("base64");
  var image = new Image();

  image.src = "data:image/gif;base64," + base64d;
  document.getElementById("setresultimg").src=image.src;

  show();
});


function show() {
  Array.prototype.forEach.call(arguments, function(elem) {
    elem.style.display = null;
  });
}

function hide() {
  Array.prototype.forEach.call(arguments, function(elem) {
    elem.style.display = 'none';
  });
}

function sendMessage(message) {
  console.log('Client sending message: ', message);
  socket.emit('message', message);
}

function sendPhoto() {
  var CHUNK_LEN = 64000;
  console.log('width and height ', photoContextW, photoContextH);
  var img = photoContext.getImageData(0, 0, photoContextW, photoContextH);
  len = img.data.byteLength;
  n = len / CHUNK_LEN | 0;

  console.log('Sending a total of ' + len + ' byte(s)');

  if (!dataChannel) {
    logError('Connection has not been initiated. ' +
      'Get two peers in the same room first');
    return;
    } else if (dataChannel.readyState === 'closed') {
    logError('Connection was lost. Peer closed the connection.');
    return;
  }


  dataChannel.send(len);

  for (var i = 0; i < n; i++) {
    var start = i * CHUNK_LEN,
    end = (i + 1) * CHUNK_LEN;
    console.log(start + ' - ' + (end - 1));
    dataChannel.send(img.data.subarray(start, end));
  }

  if (len % CHUNK_LEN) {
    console.log('last ' + len % CHUNK_LEN + ' byte(s)');
    dataChannel.send(img.data.subarray(n * CHUNK_LEN));
  }
}