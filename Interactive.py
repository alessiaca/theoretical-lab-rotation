import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button

class StupidNeuron:
    def __init__(self, name, inp = None, position = None, tau = 100, v = 0, ext = 0):
        self.position = position
        self.name = name
        self.tau = tau
        self.inputs = inp
        self.weight = 0
        self.v = 0
        self.f_rate = 0
        self.ext = ext
        self.external = 0
        
    def potential(self):
        if self.inputs != None:
            synaptic = self.inputs.f_rate * self.weight
        else: 
            synaptic = 0
        self.v += (-self.v + synaptic + self.external)/self.tau
                  
    def activation(self):
        fire = np.tanh(self.v)
        self.f_rate = fire * (fire > 0)
        
    def NeuronJob(self):
        self.potential()
        self.activation()
        
    def connect(self,weight):
        self.weight = weight
        
    def click(self):
        if self.external == 0:
            self.external = self.ext
        else:
            self.external = 0
    
def step(units):
    for unit in units:
        unit.NeuronJob()
        
def WhoIsWho(units):
    names = []
    for unit in units:
        names.append(unit.name)
    return names


class Simulation:
    def __init__(self, duration, actors):
        self.duration = duration
        self.total_length = 10000
        self.image = np.zeros([len(actors)+1,self.total_length])
        self.storage = np.zeros([len(actors)+1,self.duration])
        
    def __call__(self):
        for i in range(self.duration):
            step(actors)
            for actor in actors:
                self.storage[actor.position,i] = actor.f_rate
            self.storage[0,i] = actors[0].external      
        self.image = np.hstack((self.image, self.storage))
        self.image = self.image[:,-self.total_length:]
        return self.image
        

SN1 = StupidNeuron('SN1', position = 1, ext = 1)
SN2 = StupidNeuron('SN2', inp = SN1, position = 2)
W = 1
actors = [SN1,SN2]
names = ['Input']
names = names + WhoIsWho(actors)
SN2.connect(W)
simulation = Simulation(500,actors)

fig = plt.figure(figsize=(16,16))
ax = fig.add_axes([0.045,0.59,0.45,0.4])
ax.set_yticks(np.arange(len(names)))
ax.set_yticklabels(names)


########################BUTTON#########################
button_Input = fig.add_axes([0.37, 0.4, 0.1, 0.04])
Input = Button(button_Input, 'Input', color='#f7efff', hovercolor='#f7efff')
def Input_event(event):
    SN1.click()
    if SN1.external == 1:
        Input.color = '#f2c99f'
        Input.hovercolor = '#f2c99f'
    else:
        Input.color = '#f7efff'
        Input.hovercolor = '#f7efff'
Input.on_clicked(Input_event)


########################SLIDER#########################
default = W
delta = W*0.01
slider_ax = fig.add_axes([0.045, 0.4, 0.1, 0.04])
slider = Slider(slider_ax,'Weight',W*0.01, \
                         W*2, valinit = default, valstep = delta)
def update_slider(val):
   SN2.weight = slider.val
slider.on_changed(update_slider)


########################ANIMATION#########################
updater = ax.imshow(simulation(), aspect = 'auto', vmin=0, vmax=1)
def updatefig(*args):
    v = simulation()
    updater.set_array(v)
    return updater,
ani = animation.FuncAnimation(fig, updatefig, interval=200, blit=True)
plt.show()
