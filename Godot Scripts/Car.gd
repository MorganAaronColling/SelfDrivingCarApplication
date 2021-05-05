extends VehicleBody

############################################################
# behaviour values
export var SELF_DRIVING = true
export var MAX_ENGINE_FORCE = 100
export var MAX_BRAKE_FORCE = 5.0
export var MAX_STEER_ANGLE = 0.5

export var steer_speed = 5.0

var steer_target = 0.0
var steer_angle = 0.0
var steer_val = 0

############################################################
# Input

export var joy_steering = JOY_ANALOG_LX
export var steering_mult = -1.0
export var joy_throttle = JOY_ANALOG_R2
export var throttle_mult = 1.0
export var joy_brake = JOY_ANALOG_L2
export var brake_mult = 1.0
onready var headlight_R = $Headlight_R
onready var headlight_L = $Headlight_L

func _ready():
	friction = 0.5
	pass
	
func _physics_process(delta):
	if abs(Input.get_joy_axis(0, joy_steering)) >=0.05:
		steer_val = steering_mult * Input.get_joy_axis(0, joy_steering)
	else:
		steer_val = 0
	
	
	var throttle_val = throttle_mult * Input.get_joy_axis(0, joy_throttle)
	var brake_val = brake_mult * Input.get_joy_axis(0, joy_brake)
	
	# overrules for keyboard
	if Input.is_action_pressed("ui_up"):
		throttle_val = 1.0
	if Input.is_action_pressed("ui_down"):
		brake_val = 1.0
	if Input.is_action_pressed("ui_left"):
		steer_val = 1.0
	elif Input.is_action_pressed("ui_right"):
		steer_val = -1.0
	
	headlight_change()
	toggle_self_drive()
	return_to_menu()
		
	engine_force = throttle_val * MAX_ENGINE_FORCE
	brake = brake_val * MAX_BRAKE_FORCE
	
	steer_target = steer_val * MAX_STEER_ANGLE
	if (steer_target < steer_angle):
		steer_angle -= steer_speed * delta
		if (steer_target > steer_angle):
			steer_angle = steer_target
	elif (steer_target > steer_angle):
		steer_angle += steer_speed * delta
		if (steer_target < steer_angle):
			steer_angle = steer_target
	
	if SELF_DRIVING:
		steering = Autoload.steeringAngle
		
	else:
		steering = steer_angle
		Autoload.steeringAngle = steer_angle
	
	Autoload.ui_steering = steering
	Autoload.ui_engine = engine_force
	Autoload.ui_brake_force = brake


func headlight_change():
		if Input.is_action_just_pressed("headlights"):
			if headlight_L.visible == true:
				headlight_L.visible = false
				headlight_R.visible = false
			else:
				headlight_L.visible = true
				headlight_R.visible = true

func toggle_self_drive():
	if Input.is_action_just_pressed("toggle_self_drive"):
		if SELF_DRIVING == true:
			SELF_DRIVING = false
		else:
			SELF_DRIVING = true

func return_to_menu():
	if Input.is_action_just_pressed("main_menu"):
		get_tree().change_scene("res://Scenes/Main_Menu.tscn")
