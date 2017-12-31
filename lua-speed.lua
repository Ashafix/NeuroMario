--joypad.setfrommnemonicstr('|.S|............|............|')
--savestate.loadslot(1)
index = 0
while true do
	joypad_input = tostring(client.neuralnet())

	if (index > 12000) or (joypad_input == 'Restart') then
		index = 0
		savestate.loadslot(1)
	else
		joypad.setfrommnemonicstr(joypad_input)
		emu.frameadvance()
		index = index + 1
	end
end
