--client.openrom('D:\Users\Ashafix\Documents\GitHub\NeuroMario\Super Mario Kart (USA).zip')

--joypad.setfrommnemonicstr('|.S|............|............|')
--savestate.loadslot(1)
index = 0
--client.SocketServerConnect()
i = 0
while i < 100 do
	--print('sending screenshot')
	client.SocketServerScreenShot()
	print(client.SocketServerSuccessful())
	print(client.SocketServerSuccessful() == "False")
	if (client.SocketServerSuccessful() == "False") then
		client.SocketServerScreenShot()
		i = 10000
	else
		emu.frameadvance()
	end
	--print('retrieving input')
	--joypad_input = tostring(client.SocketServerResponse())
	--joypad_input = '|..|............|............|'

	--if (index > 100) or (joypad_input == 'Restart') then
--		index = 0
	--	savestate.loadslot(1)
	--else
--		joypad.setfrommnemonicstr(joypad_input)
	--	emu.frameadvance()
--		index = index + 1
	--end
	i = i + 1
end
print(i)
