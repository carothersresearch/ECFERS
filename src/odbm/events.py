import roadrunner as rr

def addEvent(model, label, event, trigger):
    '''

    event can be a list of events 
    '''
    ''' 
    ex:
    addEvent(label = 'dna', event = 'dna_1 = 1, dna_2 = 2', trigger =  't > 10') 
    '''
    #To-do:
    # check uniqueness of event label
    
    events = event.replace(' ','').split(',')
    labels = label.replace(' ','').split(',')
    for idx, ev in enumerate(events):
        var = ev.split('=')[0]
        form = ev.split('=')[1]
        model.addEvent(labels[idx], False, trigger, False)
        model.addEventAssignment(eid = labels[idx], vid = var, formula = form, forceRegenerate = False)

def removeEvent(model, label):
    labels = label.replace(' ','').split(',')
    for id in labels:
        model.removeEvent(id, False)

def forceRegenerate(model):
    model.addEvent('dummy', False, 'time > 0 ', False)
    model.removeEvent('dummy', forceRegenerate = True)


'''
from Tellurium docs:
    User passes in, for example:
    'at (time > 10): X = sin(t)'
    trigger is left of ':': time > 10
    event is right of ':': X = sin(t)


    Examples from Tellurium:
        at sin(2*pi*time/period) >  0, t0=false: UpDown = 1
        at sin(2*pi*time/period) <= 0, t0=false: UpDown = 0
        at (x>5): y=3, x=r+2;
        E1: at(x>=5): y=3, x=r+2;
'''