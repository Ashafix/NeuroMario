index = 0

i = 0
savestate.loadslot(1)
emu.frameadvance()

while i < 10000 do
	--print(i)
	joypad_input = tostring(comm.httppostscreenshot())
	--print(joypad_input)
	if (joypad_input == 'Restart') then
		savestate.loadslot(1)
		i = 0
	else
		joypad.setfrommnemonicstr(joypad_input)
	end
	emu.frameadvance()
	i = i + 1
	
end

