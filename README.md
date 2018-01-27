# NeuroMario

NeuroMario is a Python machine learning module for a self-driving, self-learning AI playing Super Mario Kart on a SNES emulator.

With a very limited input (5 human runs with 5 rounds each, ~10 min of training data), a neural net based on Nvidia's self-driving car publication [1] was trained and able to drive autonomously.
The AI has only access to the images produced by the emulator and not internal values such as position or speed.
Using SciPy Differential Evolution optimization strategy [2] the neural net was optimized to achieve run times which were superior to the training data.

Additional requirements:
* BizHaw emulator (based on this pull request: https://github.com/TASVideos/BizHawk/pull/1080)
* A Super Mario Kart ROM (see: http://www.retrode.org/)

Other features include:
* Automatic detection of passing the finish line (based on a Random Decision Forest taking a screenshot as the input)
* Reading run times from screen shots (based on a Support Vector Machine)
* Parallelization of the learning possible
* Client-server structure, AI can be run a separate machine

The module was written for Super Mario Kart but most functions are applicable to other SNES games as well.

## Architecture

* [https://github.com/TASVideos/BizHawk-Prereqs/releases](Bizhawk/Emuhawk) is used for emulating the SNES console
* A Python server (either socket, HTTP or memory mapped file based) retrieves screenshots
* The screenshots are evaluated and a response is sent back to the emulator
* [http://tasvideos.org/Bizhawk/LuaFunctions.html](Lua) is used as a scripting language in Emuhawk

References:

1: Bojarski et al.: End to End Learning for Self-Driving Cars, [https://arxiv.org/abs/1604.07316](arXiv:1604.07316)<br />
see also: https://github.com/kevinhughes27/TensorKart and https://github.com/0bserver07/Nvidia-Autopilot-Keras<br />
2: https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.optimize.differential_evolution.html
