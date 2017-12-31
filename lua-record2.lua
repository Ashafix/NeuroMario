index = 0

i = 0

savestate.loadslot(2)
while i < 8000 do
	
	joypad_input = tostring(comm.httppostscreenshot())
	if joypad_input == 'Restart' then
		i = 0
		savestate.loadslot(2)
	else
		joypad.setfrommnemonicstr(joypad_input)
		emu.frameadvance()
	end
	i = i + 1
	
end

client.shutdown()
