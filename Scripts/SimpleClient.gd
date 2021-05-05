extends Node

var socket
var paused = true
signal pause
signal unpause
onready var steeringAngle = PoolByteArray(str(Autoload.steeringAngle).to_utf8())

func _init():
	socket = PacketPeerUDP.new()
	socket.set_dest_address("127.0.0.1", 5005)
	
func _process(delta):
	steeringAngle = PoolByteArray(str(Autoload.steeringAngle).to_utf8())
	is_paused()


func is_paused():
	if Input.is_action_just_pressed("pause_data"):
		if paused == true:
			paused = false
			emit_signal("unpause")
		else:
			paused = true
			emit_signal("pause")
	
	
func _on_Timer_timeout():
	if paused == false:
		socket.put_packet(steeringAngle)
	else:
		socket.put_packet(PoolByteArray(str("paused").to_utf8()))
		pass
