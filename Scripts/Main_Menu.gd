extends MarginContainer

func _on_Train_1_pressed():
	get_tree().change_scene("res://Train_Tracks/Train_Track_1.tscn")


func _on_Train_2_pressed():
	get_tree().change_scene("res://Train_Tracks/Train_Track_2.tscn")


func _on_Train_3_pressed():
	get_tree().change_scene("res://Train_Tracks/Train_Track_3.tscn")


func _on_Test_1_pressed():
	get_tree().change_scene("res://Test_Tracks/Test_Track_1.tscn")


func _on_Test_2_pressed():
	get_tree().change_scene("res://Test_Tracks/Test_Track_2.tscn")


func _on_Test_3_pressed():
	get_tree().change_scene("res://Test_Tracks/Test_Track_3.tscn")
