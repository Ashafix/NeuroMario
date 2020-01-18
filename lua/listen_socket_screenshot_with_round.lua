print('listening to socket')
snes.setlayer_obj_1(true)
snes.setlayer_obj_2(true)
snes.setlayer_obj_3(true)
snes.setlayer_obj_4(true)
snes.setlayer_bg_1(true)
snes.setlayer_bg_2(true)
snes.setlayer_bg_3(true)
snes.setlayer_bg_4(true)

start_round = memory.readbyte(0x0010F9) + 1

index = 0
while index < 20000 do
	current_round = memory.readbyte(0x0010F9)
	comm.socketServerScreenShot()
	resp = ''
	retries = 1000
	while resp == '' do
		resp = comm.socketServerResponse()
		retries = retries - 1
		if retries == 0 then
			resp = 'failed'
			index = 100000
		end
	end
	
	while string.len(resp) >= 30 do
		index = index + 1
		command = string.sub(resp, 1, 30)
		joypad.setfrommnemonicstr(command)
		emu.frameadvance()
		resp = string.sub(resp, 31)
	end
	if current_round > start_round then
		minutes = math.floor(tonumber(memory.readbyte(0x000104)) / 16) * 10 + tonumber(memory.readbyte(0x000104)) % 16
		seconds = math.floor(tonumber(memory.readbyte(0x000102)) / 16) * 10 + tonumber(memory.readbyte(0x000102)) % 16
		centisecs = math.floor(tonumber(memory.readbyte(0x000101)) / 16) * 10 + tonumber(memory.readbyte(0x000101)) % 16
		
		comm.socketServerSend(tostring(minutes*60*100 + seconds*100 + centisecs))
		comm.socketServerSend(string.char(0x82))
		client.exit()
	end
end







