extends Node
var socket
var data

func _init():
	socket = PacketPeerUDP.new()
	if(socket.listen(1234,"127.0.0.1") != OK):
		print("An error occurred listening on port 1234")
	else:
		print("Listening on port 1234 on localhost")
		
func _process(delta):
	if(socket.get_available_packet_count() > 0):
		data = socket.get_packet().get_string_from_utf8()
		print(float(data))
		Autoload.steeringAngle = float(data)
