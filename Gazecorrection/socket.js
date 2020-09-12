'use strict';
var zeropc = require("zerorpc");
var client = new zeropc.Client();

const SocketIO = require('socket.io');
const axios = require('axios');

//let {PythonShell} = require('python-shell');
//var fs = require("fs");
var clientnum = 0;
var w_gaze = new Array();
w_gaze[0] = -0.25;
w_gaze[1] = -0.25;
var horizontalVec = new Array();
horizontalVec[0] = 0;
horizontalVec[1] = 0;
var users = {
  'test': {
      id: 'test',
      pw: 'test'
  }
}; // 기본 회원이 담기는 object
var onlineUsers = {}; // 현재 online인 회원이 담기는 object

module.exports = (server, app, sessionMiddleware) => {
  const io = SocketIO(server, { path: '/socket.io' });

  app.set('io', io);
  const room = io.of('/room');
  const chat = io.of('/chat');
  const setting = io.of('/setting');

  io.use((socket, next) => {
    sessionMiddleware(socket.request, socket.request.res, next);
  });

  io.on('connection', (socket) => {
    console.log('main connection' + socket.id);
    
    socket.on("join user", function (data, cb) {
      if (joinCheck(data)) {
          cb({
              result: false,
              data: "이미 존재하는 회원입니다."
          });
          return false;
      } else {
          users[data.id] = {
              id: data.id,
              pw: data.pw,
          };
          cb({
              result: true,
              data: "회원가입에 성공하였습니다."
          });

      }
  });

  socket.on("login user", function (data, cb) {
      if (loginCheck(data)) {
          onlineUsers[data.id] = {
              roomId: 1,
              socketId: socket.id
          };
          socket.join('room1');
          cb({
              result: true,
              data: "로그인에 성공하였습니다."
          });
          updateUserList(0, 1, data.id);
      } else {
          cb({
              result: false,
              data: "등록된 회원이 없습니다. 회원가입을 진행해 주세요."
          });
          return false;
      }
  });

  socket.on('logout', function () {
      if (!socket.id) return;
      let id = getUserBySocketId(socket.id);
      let roomId = onlineUsers[id].roomId;
      delete onlineUsers[getUserBySocketId(socket.id)];
      updateUserList(roomId, 0, id);
  });

  socket.on('disconnect', function () {
      console.log('main disconnect');
      if (!socket.id) return;
      let id = getUserBySocketId(socket.id);
      if(id === undefined || id === null){
          return;
      }
      let roomId = onlineUsers[id].roomId || 0;
      delete onlineUsers[getUserBySocketId(socket.id)];
      updateUserList(roomId, 0, id);
  });

  socket.on('join room', function (data) {
      let id = getUserBySocketId(socket.id);
      let prevRoomId = onlineUsers[id].roomId;
      let nextRoomId = data.roomId;
      socket.leave('room' + prevRoomId);
      socket.join('room' + nextRoomId);
      onlineUsers[id].roomId = data.roomId;
      updateUserList(prevRoomId, nextRoomId, id);
  });

  function updateUserList(prev, next, id) {
      if (prev !== 0) {
          io.sockets.in('room' + prev).emit("userlist", getUsersByRoomId(prev));
          io.sockets.in('room' + prev).emit("lefted room", id);
      }
      if (next !== 0) {
          io.sockets.in('room' + next).emit("userlist", getUsersByRoomId(next));
          io.sockets.in('room' + next).emit("joined room", id);
      }
  }

  function loginCheck(data) {
      if (users.hasOwnProperty(data.id) && users[data.id].pw === data.pw) {
          return true;
      } else {
          return false;
      }
  }

  function joinCheck(data) {
      if (users.hasOwnProperty(data.id)) {
          return true;
      } else {
          return false;
      }
  }

  function getUserBySocketId(id) {
      return Object.keys(onlineUsers).find(key => onlineUsers[key].socketId === id);
  }

  function getUsersByRoomId(roomId) {
      let userstemp = [];
      Object.keys(onlineUsers).forEach((el) => {
          if (onlineUsers[el].roomId === roomId) {
              userstemp.push({
                  socketId: onlineUsers[el].socketId,
                  name: el
              });
          }
      });
      return userstemp;
  }
  });
  /*
  function assignwgaze(value, id){
    users[id] = {
      wgaze: value
    };
    console.log('id, wgaze:' + id);
  }
  function assignpupil(value, id){
    users[id] = {
      pupil: value
    };
    console.log('id, pupil:' + id +',' + pupil);
  }*/
  
  function assignwgaze(value, clientnum)
  {
    w_gaze[clientnum] = value;
  }
  function assignpupil(value, clientnum)
  {
    horizontalVec[clientnum] = value;
  }
  function getUserBySocketId(id) {
    return Object.keys(users).find(key => users[key].socketId === id);
  }
  setting.on('connection', (socket) => {
    console.log('setting 네임스페이스 접속' + socket.id);
    socket.on('wgaze', (value) => {
      assignwgaze(value, clientnum);
      //let id = getUserBySocketId(socket.id);
      //assignwgaze(value, id);
    })
    socket.on('pupil', (value) => {
      assignpupil(value, clientnum);
      //let id = getUserBySocketId(socket.id);
      //assignpupil(value, id);
    })
    socket.emit('GetClientNum', clientnum);

    client.connect("tcp://127.0.0.1:4242");
    socket.on('send', function(blob, clientnumber){
        var base64data = blob.toString('base64');
        client.invoke("mainCorrection", base64data, w_gaze[clientnumber], horizontalVec[clientnumber], clientnumber, function(error,res,more){
          if(res!=undefined)
            socket.emit('get',res);
        });
    });
  })

  room.on('connection', (socket) => {
    console.log('room 네임스페이스에 접속' + socket.id);
    socket.on('disconnect', () => {
      console.log('room 네임스페이스 접속 해제');
    });
  });
  
  chat.on('connection', (socket) => {
    console.log('chat 네임스페이스에 접속' + socket.id);
    const req = socket.request;
    const { headers: { referer } } = req;
    const roomId = referer
      .split('/')[referer.split('/').length - 1]
      .replace(/\?.+/, '');
    socket.join(roomId);
    socket.to(roomId).emit('join', {
      user: 'system',
      //chat: `${req.session.color}님이 입장하셨습니다.`,
      chat: '새로운 참가자가 입장했습니다.',//추가
    });
    
    //webrtc추가
    //클라이언트 구분
    console.log(clientnum);
    socket.emit('GetClientNum', clientnum);

    // 선착순 2명씩 들어올 경우로 가정

    clientnum++;

    socket.emit('roomid', roomId);//client에게 roomid정보전달
    socket.on('message', function(message) {
      console.log('Client said: ', message);
      // for a real app, would be room-only (not broadcast)
      socket.broadcast.emit('message', message);
    });
    socket.on('create or join', function(room) {
      room = roomId;
      console.log('Received request to create or join room ' + room);
      const currentRoom = socket.adapter.rooms[roomId];
      const userCount = currentRoom ? currentRoom.length : 0;
      //var clientsInRoom = io.sockets.adapter.rooms[room];
      //var numClients = clientsInRoom ? Object.keys(clientsInRoom.sockets).length : 0;
      console.log('Room ' + room + ' now has ' + userCount + ' client(s)');
  
      if (userCount === 1) {
        //socket.join(room);
        console.log('Client ID ' + socket.id + ' created room ' + room);
        socket.emit('created', room, socket.id);
  
      } else if (userCount === 2) {
        console.log('Client ID ' + socket.id + ' joined room ' + room);
        io.sockets.emit('join', room);
        //ssocket.join(room);
        socket.emit('joined', room, socket.id);
        io.sockets.emit('ready');
      } 
    });
  
    socket.on('ipaddr', function() {
      var ifaces = os.networkInterfaces();
      for (var dev in ifaces) {
        ifaces[dev].forEach(function(details) {
          if (details.family === 'IPv4' && details.address !== '127.0.0.1') {
            socket.emit('ipaddr', details.address);
          }
        });
      }
    });
  
    socket.on('bye', function(){
      console.log('received bye');
    });
  
    client.connect("tcp://127.0.0.1:4242");
    socket.on('send', function(blob, clientnumber){
        var base64data = blob.toString('base64');
        client.invoke("mainCorrection", base64data, w_gaze[clientnumber], horizontalVec[clientnumber], clientnumber, function(error,res,more){
          if(res!=undefined)
          {
            socket.to(roomId).emit('get', res, clientnumber);
            socket.emit('get',res, clientnumber);
          }
        });
    });
    
    socket.on('disconnect', () => {
      console.log('chat 네임스페이스 접속 해제');
      socket.leave(roomId);
      const currentRoom = socket.adapter.rooms[roomId];
      const userCount = currentRoom ? currentRoom.length : 0;
      if (userCount === 0) {
        axios.delete(`http://localhost:8005/room/${roomId}`)
          .then(() => {
            console.log('방 제거 요청 성공');
          })
          .catch((error) => {
            console.error(error);
          });
      } else {

        socket.to(roomId).emit('exit', {
          user: 'system',
          chat: `${req.session.color}님이 퇴장하셨습니다.`,
        });
      }
    });
  });
};
