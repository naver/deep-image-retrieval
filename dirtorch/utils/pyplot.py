''' Just a bunch of pyplot utilities...
'''
import pdb
import numpy as np


def pyplot(globs=None, ion=True, backend='TkAgg'): #None):
    ''' load pyplot and shit, in interactive mode
    '''
    globs = globs or globals()
    if 'pl' not in globs:
        if backend:
            import matplotlib
            matplotlib.use(backend)
        import matplotlib.pyplot as pl
        if ion: pl.ion()
        globs['pl'] = pl


def figure(name, clf=True, **kwargs):
    pyplot()
    f = pl.figure(name, **kwargs)
    f.canvas.manager.window.attributes('-topmost',1)
    pl.subplots_adjust(0,0,1,1,0,0)
    if clf: pl.clf()
    return f


def pl_imshow( img, **kwargs ):
    if isinstance(img, str):
        from PIL import Image
        img = Image.open(img)
    pyplot()
    pl.imshow(img, **kwargs)


def pl_noticks():
    pl.xticks(())
    pl.yticks(())

def fig_imshow( figname, img, **kwargs ):
    fig = figure(figname)
    pl_imshow( img, ** kwargs)
    pdb.set_trace()


def crop_text(sentence, maxlen=10):
    lines = ['']
    for word in sentence.split():
        t = lines[-1] + ' ' + word
        if len(t) <= maxlen:
            lines[-1] = t
        else:
            lines.append( word )
    if lines[0] == '': lines.pop(0)
    return lines


def plot_bbox( bbox, fmt='rect', color='blue', text='', text_effects=False, text_on_box=False, scale=True, fill_color=None, ax=None, **kwargs ):
    pyplot()
    ax = ax or pl.gca()
    
    if fmt == 'rect' or fmt == 'xyxy':
        ''' bbox = (left, top, right, bottom)
        '''
        assert len(bbox) == 4, pdb.set_trace()
        x0,y0,x1,y1 = bbox
        X,Y = [x0,x0,x1,x1,x0], [y0,y1,y1,y0,y0]
        
    elif fmt == 'box' or fmt == 'xywh':
        ''' bbox = (left, top, width, height)
        '''
        assert len(bbox) == 4, pdb.set_trace()
        x0,y0,w,h = bbox
        X,Y = [x0,x0,x0+w,x0+w,x0], [y0,y0+h,y0+h,y0,y0]
    
    elif fmt == '4pts':
        ''' bbox = ((lx,ly), (tx,ty), (rx,ty), (bx,by))
        '''
        assert len(bbox) >= 4, pdb.set_trace()
        bbox = np.asarray(bbox)
        X, Y = bbox[:,0], bbox[:,1]
        X = list(X)+[X[0]]
        Y = list(Y)+[Y[0]]

    elif fmt == '8val':
        ''' bbox = 8-tuples of (x0,y0, x1,y0, x0,y1, x1,y1)
        '''
        assert len(bbox) >= 8, pdb.set_trace()
        X, Y = bbox[0::2], bbox[1::2]
        
    else: 
        raise ValueError("bad format for a bbox: %s" % fmt)

    ls = kwargs.pop('ls','-')
    line = ax.plot( X, Y, ls, scalex=scale, scaley=scale, color=color, **kwargs)

    if fill_color:
        ax.fill(X, Y, fill_color, alpha=0.3)
    if text:
        text = '\n'.join(crop_text(text, 10))
        
        color = line[0].get_color()
        
        if text_on_box:
            text = ax.text(bbox[0], bbox[1]-2, text, fontsize=8, color=color)
        else:
            text = ax.text( np.mean(X), np.mean(Y), text,
                            ha='center', va='center', fontsize=16, color=color,
                            clip_on=True)
            
        if text_effects:
            import matplotlib.patheffects as path_effects
            effects = [path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal()]
            text.set_path_effects(effects)
        
    return line

def plot_rect( rect, **kwargs):
    return plot_bbox( rect, fmt='xyxy', **kwargs )

def plot_poly( poly, **kwargs ):
    return plot_bbox( poly, fmt='4pts', **kwargs )


def plot_cam_on_map( pose, fmt="xyz,rpy", fov=70.0, cone=10, marker='+', color='r', ax=None):
    if not ax: ax = pl.gca()
    
    if fmt == "xyz,rpy":
        x,y = pose[0,0:2]
        A = pose[1,2] + fov*np.pi/180/2
        B = pose[1,2] - fov*np.pi/180/2
    elif fmt == "xylr":
        x,y,A,B = pose
    else:
        raise ValueError("Unknown pose format %s" % str(fmt))
    
    ax.plot(x,y, marker, color=color)
    ax.plot([x,x+cone*np.cos(A)],[y,y+cone*np.sin(A)], '--', color=color)
    ax.plot([x,x+cone*np.cos(B)],[y,y+cone*np.sin(B)], '--', color=color)


def subplot_grid(nb, n, aspect=1):
    """ automatically split into rows and columns
    
    aspect : float. aspect ratio of the subplots (width / height).
    """
    pyplot()
    nr = int(np.sqrt(nb * aspect))
    nc = int((nb-1) / nr + 1)
    return pl.subplot(nr, nc, n)




























