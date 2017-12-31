slot_number = 1

index = 0
savestate.loadslot(slot_number)

repeats = 6
while index < 8000 do
	
	joypad_input = tostring(comm.httppostscreenshot())
	if joypad_input == 'Restart' then
		index = 0
		savestate.loadslot(slot_number)
	else
		z = 0
		while z < repeats do
			joypad.setfrommnemonicstr(joypad_input)
			emu.frameadvance()
			z = z + 1
		end
		index = index + z
	end
end

client.exit()
