extends Spatial

onready var cameracar = $Car/CameraCar
onready var cameraabovecar = $CameraAboveCar
onready var environment = $WorldEnvironment.environment
onready var day_env = preload("res://Environments/sky_day.tres")
onready var night_env = preload("res://Environments/sky_night.tres")
var day = true

func _ready():
	cameracar.make_current()
	
func _process(delta):
	camera_change()
	environment_change()

	
func camera_change():
	if Input.is_action_just_released("camera_change") and cameraabovecar.is_current():
			cameracar.make_current()
	elif Input.is_action_just_released("camera_change") and cameracar.is_current():
			cameraabovecar.make_current()

func environment_change():
	if Input.is_action_just_pressed("environment_change"):
		if day == true:
			print("setting night")
			environment.background_sky = night_env
			day = false
		else:
			print("setting day")
			environment.background_sky = day_env
			day = true
			


func _on_Client_pause():
	$UI/Label.visible = true


func _on_Client_unpause():
	
	$UI/Label.visible = false


