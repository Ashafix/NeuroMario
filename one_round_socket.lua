print('One round')
comm.socketServerScreenShot()

while true do
	comm.socketServerScreenShot()
	resp = ''
	while resp == '' do
		resp = comm.socketServerResponse()
	end
	while string.len(resp) >= 30 do
		command = string.sub(resp, 1, 30)
		joypad.setfrommnemonicstr(command)
		emu.frameadvance()
		resp = string.sub(resp, 31)
	end
end


client.exit()
