index = 0

while index < movie.length() do
	emu.frameadvance()
	index = index + 1
end

savestate.save("unittest.state")
client.exit()
