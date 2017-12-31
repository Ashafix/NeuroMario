index = 0

while index < movie.length():
	emu.frameadvance()
	index = index + 1


savestate.save("DefinedState.state")
