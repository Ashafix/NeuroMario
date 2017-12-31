index = 0
comm.httpPost(comm.httpGetPostUrl(), index)
running = false
while running do
	resp = ''
	comm.httpPost(comm.httpGetPostUrl(), index)
	while resp == '' do
		resp = comm.httpGet(comm.httpGetGetUrl())
	end
	print(resp)
	if resp == 'screenshot' then
		running = false
		comm.socketServerScreenShot()
	end
	while string.len(resp) >= 30 do
		index = index + 1
		command = string.sub(resp, 1, 30)
		joypad.setfrommnemonicstr(command)
		emu.frameadvance()
		resp = string.sub(resp, 31)
	end
end


--client.exit()
