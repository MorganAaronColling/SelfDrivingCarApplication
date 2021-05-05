extends CanvasLayer

onready var engine_force = round(Autoload.ui_engine)
onready var angle = Autoload.ui_steering
onready var brake_force = round(Autoload.ui_brake_force * 20)

func _ready():
		$Data.text = (" |Engine Force: " + str(engine_force) + "\n" + " |Break Force: "
		+ str(brake_force) + "\n" + "|Steering Angle: " + str(angle))


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	engine_force = round(Autoload.ui_engine)
	angle = Autoload.ui_steering
	brake_force = round(Autoload.ui_brake_force * 20)
	$Data.text = (" |Engine Force: " + str(engine_force) + "\n" + " |Break Force: "
		+ str(brake_force) + "\n" + " |Steering Angle: " + str(angle))
