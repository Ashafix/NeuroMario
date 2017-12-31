print('hello')
--comm.socketServerSetTimeout(25)
--emu.frameadvance()
index = 0
comm.socketServerScreenShot()
running = true
while running do
	
	comm.socketServerSend(index)
	resp = ''
	while resp == '' do
		resp = comm.socketServerResponse()
	end
	if resp == 'screenshot' then
		running = false
		comm.socketServerScreenShot()
	end
	while string.len(resp) >= 30 do
		index = index + 1
		--print(string.len(resp))
		command = string.sub(resp, 1, 30)
		joypad.setfrommnemonicstr(command)
		emu.frameadvance()
		resp = string.sub(resp, 31)
	end
end


client.exit()
