
//BASIC
var zerorpc = require("zerorpc");

var client = new zerorpc.Client();
client.connect("tcp://127.0.0.1:8080");



//이런식으로 python 함수 사용
client.invoke("hello","RPC",function(error,res,more){
    console.log(res);
});


//스트리밍 RESPONSE
client.invoke("streaming_range", 10,20,2,funcion(error,res,more){
    console.log(res);
});
