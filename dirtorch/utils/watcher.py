"""Watch is an object to easily watch what is happening
during the training/evaluation of a deep net.
"""
import os
import time
import pdb
from collections import defaultdict
import numpy as np

from . import evaluation


class AverageMeter(object):
    """Computes and stores the average and current value of a metric.
    It is fast (constant time), regardless of the lenght of the measure series.

    mode: (str). Behavior of the meter.
          'average': just the average of all values since the start
          'sliding': just the average of the last 'nlast' values
          'last':    just the last value (=='sliding' with nlast=1)
          'min' :    the minimum so far
          'max' :    the maximum so far
    """

    def __init__(self, mode='average', nlast=5):
        self.mode = mode
        self.nlast = nlast
        self.reset()

    def reset(self):
        self.vals = []
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.is_perf = False

    def export(self):
        return {k:val for k,val in self.__dict__.items() if type(val) in (bool, str, float, int, list)}

    def update(self, val, weight=1):
        ''' sliding window average '''
        self.vals.append( val )
        self.sum += val * weight
        self.count += weight
        if self.mode == 'average':
            self.avg = self.sum / self.count
        elif self.mode == 'sliding':
            vals = self.vals[-self.nlast:]
            self.avg = sum(vals) / (1e-8+len(vals))
        elif self.mode == 'last':
            self.avg = val
        elif self.mode == 'min':
            self.avg = min(self.avg or float('inf'), val)
        elif self.mode == 'max':
            self.avg = max(self.avg or -float('inf'), val)
        else:
            raise ValueError("unknown AverageMeter update policy '%s'" % self.mode)

    def __bool__(self):
        return bool(self.count)
    __nonzero__ = __bool__    # for python2

    def __len__(self):
        return len(self.vals)

    def tostr(self, name='', budget=100, unit=''):
        ''' Print the meter, using more or less characters
        '''
        _budget = budget
        if name:
            name += ': '
            budget -= len(name)

        if isinstance(self.avg, int):
            avg = '%d' % self.avg
            minavg = len(avg)
            val = ''
            budget -= len(avg) + len(unit)
        else:
            avg = '%f' % self.avg
            minavg = (avg+'.').find('.')

            val = 'last: %f' % self.vals[-1]
            minval = (val+'.').find('.')

            budget -= len(avg) + len(val) + 3 + 2*len(unit)

        while budget < 0 :
            old_budget = budget

            if len(val):
                val = val[:-1]
                budget += 1
                if len(val) < minval:
                    val = '' # we cannot delete beyond the decimal point
                    budget += 3 + len(val) + len(unit) # add parenthesis
                    continue
                else:
                    if len(val) % 2: continue # shrink the other sometimes

            if len(avg) >= minavg and len(name) <= len(avg):
                avg = avg[:-1]
                budget += 1
                continue # can shrink further

            if len(name) > 2:
                name = name[:-2]+' ' # remove last char
                budget += 1

            # cannot shrink anymore
            if old_budget == budget: break

        res = name + avg+unit
        if val: res += ' (' + val+unit + ')'
        res += ' '*max(0, len(res) - _budget)
        return res


class Watch (object):
    """ 
    Usage:
    ------
      - call start() just before the loop
      - call tic() at the beginning of the loop (first line)
      - call eval_train(measure1=score1, measure2=score2, ...) or eval_test(...)
      - call toc() and the end of the loop (last line)
      - call stop() after the loop

    Arguments:
    ----------
    tfreq: (float or None)
            temporal frequency of outputs (in seconds)

    nfreq: (int or None)
            iteration frequency of outputs (in iterations)
    """
    def __init__(self, tfreq=30.0, nfreq=None):
        self.tfreq = tfreq
        self.nfreq = nfreq

        # init meters
        self.meters = defaultdict(AverageMeter)
        self.meters['epoch'] = AverageMeter(mode='last')
        self.meters['test_epoch'] = AverageMeter(mode='last')
        self.meters['data_time'] = AverageMeter(mode='sliding')
        self.meters['batch_time'] = AverageMeter(mode='sliding')
        self.meters['lr'] = AverageMeter(mode='sliding')
        self.meters['loss'] = AverageMeter(mode='sliding')

        # init current status
        self.tostr_t = None
        self.cur_n = None
        self.batch_size = None
        self.last_test = 0
        self.viz = False

    def __getattr__(self, name):
        meters = object.__getattribute__(self, 'meters')
        if name in meters:
            return meters[name]
        else:
            return object.__getattribute__(self, name)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def start(self):
        '''Just before the loop over batches
        '''
        self.last_time = time.time()
        self.cur_n = 0
        self.tostr_t = self.last_time

    def tic(self, batch_size, epoch=0, **kw):
        '''Just after loading one batch
        '''
        assert self.last_time is not None, "you must call start() before the loop!"
        self.meters['data_time'].update(time.time() - self.last_time)
        self.batch_size = batch_size
        self.meters['epoch'].update(epoch)
        n_epochs = len(self.meters['epoch'])
        
        for name, val in kw.items():
            self.meters[name].mode = 'last'
            self.meters[name].update(val)
            assert len(self.meters[name]) == n_epochs, "missing values for meter %s (expected %d, got %d)" % (name, n_epochs, len(self.meters[name]))

    def eval_train(self, **measures):
        n_epochs = len(self.meters['epoch'])
        
        for name, score in measures.items():
            self.meters[name].is_perf = True
            self.meters[name].update(score, self.batch_size)
            assert len(self.meters[name]) == n_epochs, "missing values for meter %s (expected %d, got %d)" % (name, n_epochs, len(self.meters[name]))

    def eval_test(self, mode='average', **measures):
        assert self.batch_size is None, "you must call toc() before; measures should concern the entire test"
        epoch = self.meters['epoch'].avg
        self.meters['test_epoch'].update(epoch)
        n_epochs = len(self.meters['test_epoch'])

        for name, val in measures.items():
            name = 'test_'+name
            if name not in self.meters:
                self.meters[name] = AverageMeter(mode=mode)
            self.meters[name].is_perf = True
            self.meters[name].update(val)
            assert len(self.meters[name]) == n_epochs, "missing values for meter %s (expected %d, got %d)" % (name, n_epochs, len(self.meters[name]))

        if self.viz: self.plot()

    def toc(self):
        '''Just after finishing to process one batch
        '''
        assert self.batch_size is not None, "you must call tic() at the begining of the loop"

        now = time.time()
        self.meters['batch_time'].update(now - self.last_time)

        if (self.tfreq and now-self.tostr_t>self.tfreq) or (self.nfreq and (self.cur_n % self.nfreq) == 0):
            self.tostr_t = now
            n_meters = sum([bool(meter) for meter in self.meters.values()])
            cols = get_terminal_ncols()
            cols_per_meter = (cols - len('Time ')) / n_meters # columns per meter
            N = np.int32(np.linspace(0,cols - len('Time '), n_meters+1))
            N = list(N[1:] - N[:-1]) # this sums to the number of available columns

            tt = ''
            if self.meters['epoch']:
                tt += self.meters['epoch'].tostr('Epoch', budget=N.pop()-1)+' '
            tt += 'Time %s %s' % (
                    self.meters['data_time'].tostr('data',budget=N.pop()-1,unit='s'),
                    self.meters['batch_time'].tostr('batch',budget=N.pop(),unit='s'))
            for name, meter in sorted(self.meters.items()):
                if name in ('epoch', 'data_time', 'batch_time'): continue
                if meter: tt += ' '+meter.tostr(name, budget=N.pop()-1)
            print(tt)
            if self.viz: self.plot()

        self.batch_size = None
        self.cur_n += 1
        self.last_time = time.time()

    def stop(self):
        '''Just after all the batches have been processed
        '''
        res = ''
        for name, meter in sorted(self.meters.items()):
            if meter.is_perf:
                res += '\n * ' + meter.tostr(name)
        print(res[1:])

    def upgrade(self):
        '''Upgrade the old watcher to the latest version
        '''
        if not hasattr(self,'meters'):
            # convert old to new format
            self.meters = defaultdict(AverageMeter)
            self.meters['epoch'] = AverageMeter(mode='last')
            for i,name in enumerate('data_time batch_time lr loss top1 top5'.split()):
                try:
                    self.meters[name] = getattr(self,name)
                    if i < 4: self.meters[name].mode = 'sliding'
                    delattr(self, name)
                except AttributeError:
                    continue
            if not self.meters['epoch']:
                for i in range(self.epoch):
                    self.meters['epoch'].update(i)

        return self

    def measures(self):
        return {name:meter.avg for name,meter in self.meters.items() if meter.is_perf}

    def plot(self):
        ''' plot what happened so far.
        '''
        import matplotlib.pyplot as pl; pl.ion()
        self.upgrade()

        epochs = self.meters['epoch'].vals
        test_epochs = self.meters['test_epoch'].vals

        fig = pl.figure('Watch')
        pl.subplots_adjust(0.1,0.03,0.97,0.99)
        done = {'epoch','test_epoch'}

        ax = pl.subplot(321)
        ax.lines = []
        for name in 'data_time batch_time'.split():
            meter = self.meters[name]
            if not meter: continue
            done.add(name)
            n = len(meter.vals)
            pl.plot(epochs[:n], meter.vals, label=name)
        self.crop_plot(ymin=0)
        pl.legend()

        ax = pl.subplot(322)
        ax.lines = []
        for name in 'lr'.split():
            meter = self.meters[name]
            if not meter: continue
            done.add(name)
            n = len(meter.vals)
            pl.plot(epochs[:n], meter.vals, label=name)
        self.crop_plot(ymin=0)
        pl.legend()

        def avg(arr):
            from scipy.ndimage.filters import uniform_filter
            return uniform_filter(arr, size=max(3,len(arr)//20), mode='nearest')
        def halfc(color):
            pdb.set_trace()
            return tuple([c/2 for c in color])

        ax = pl.subplot(312)
        ax.lines = []
        for name in self.meters:
            if not name.startswith('loss'): continue
            meter = self.meters[name]
            if not meter: continue
            done.add(name)
            n = len(meter.vals)
            line = pl.plot(epochs[:n], meter.vals, ':', lw=0.5)
            ax.plot(epochs[:n], avg(meter.vals), '-', label=name, color=line[0].get_color())
        self.crop_plot()
        pl.legend()

        ax = pl.subplot(313)
        ax.lines = []
        for name in self.meters:
            if name in done: continue
            meter = self.meters[name]
            if not meter: continue
            done.add(name)
            n = len(meter.vals)
            if name.startswith('test_'):
                epochs_ = test_epochs[:n]
            else:
                epochs_ = epochs[:n]
            line = ax.plot(epochs_, meter.vals, ':', lw=0.5)
            ax.plot(epochs_, avg(meter.vals), '-', label=name, color=line[0].get_color())
        self.crop_plot()
        pl.legend()

        pl.pause(0.01) # update the figure

    def export(self):
        members = {}
        for k, v in self.__dict__.items():
            if k == 'meters':
                meters = {}
                for k1,v1 in v.items():
                    meters[k1] = v1.export()
                members[k] = meters
            else:
                members[k] = v
        return members

    @staticmethod
    def update_all(checkpoint):
        watch = Watch()
        for k,v in checkpoint.items():
            if 'meters' in k:
                meters = defaultdict(AverageMeter)
                for k1,v1 in v.items():
                    meter = AverageMeter()
                    meter.__dict__.update(v1)
                    meters[k1] = meter
                watch.__dict__[k] = meters
            else:
                watch.__dict__[k] = v
        return watch

    @staticmethod
    def crop_plot(span=0.5, ax=None, xmin=np.inf, xmax=-np.inf, ymin=np.inf, ymax=-np.inf):
        import matplotlib.pyplot as pl
        if ax is None: ax=pl.gca()
        if not ax.lines: return # nothing to do

        # set xlim to the last <span> of all data
        for l in ax.lines:
            x,y = map(np.asarray, l.get_data())
            xmin = min(xmin,np.min(x[np.isfinite(x)]))
            xmax = max(xmax,np.max(x[np.isfinite(x)]))
        xmin = xmax - span*(xmax-xmin)

        # set ylim to the span of remaining points
        for l in ax.lines:
            x,y = map(np.asarray, l.get_data())
            y = y[(x>=xmin) & (x<=xmax) & np.isfinite(y)] # select only relevant points
            if y.size == 0: continue
            ymin = min(ymin,np.min(y))
            ymax = max(ymax,np.max(y))

        try:
            ax.set_xlim(xmin,xmax+1)
            ax.set_ylim(ymin,(ymax+1e-8)*1.01)
        except ValueError:
            pass #pdb.set_trace()


class TensorBoard (object):
    """Tensorboard to plot training and validation loss and others

    .. notes::

        ```shell
        conda install -c conda-forge tensorboardx
        conda install tensorflow
        ```

    Args:
        logdir (str): path to save log
        phases (array): phases to plot, e.g., ['train', 'val']
    """
    def __init__(self, logdir, phases):
        from tensorboardX import SummaryWriter
        if not os.path.exists(logdir):
            for key in phases:
                os.makedirs(os.path.join(logdir, key))

        self.phases = phases
        self.tb_writer={}
        for key in phases:
            self.tb_writer[key] = SummaryWriter(os.path.join(logdir, key))

    def add_scalars(self, phase, watch, names):
        """ Add scalar values in watch.meters[names]
        """
        if not phase in self.phases:
            raise AttributeError('%s is unknown'%phase)

        epochs = sorted(watch.meters['epoch'].vals)
        for name in names:
            vals = sorted(watch.meters[name].vals)
            cnt = watch.meters[name].count
            for n, val in zip(epochs, vals):
                self.tb_writer[phase].add_scalar(name, val, n*cnt)

    def close():
        for key in self.phases:
            self.tb_writer[key].close()        
    

 
def get_terminal_ncols(default=160):
    try:
        import sys
        from termios import TIOCGWINSZ
        from fcntl import ioctl
        from array import array
    except ImportError:
        return default
    else:
        try:
            return array('h', ioctl(sys.stdout, TIOCGWINSZ, '\0' * 8))[1]
        except:
            try:
                from os.environ import get
            except ImportError:
                return default
            else:
                return int(get('COLUMNS', 1)) - 1



if __name__ == '__main__':
    import time

    # test printing size
    batch_size = 256

    watch = Watch(tfreq=0.5)
    watch.start(epoch=0)
    watch.meters['top1'].is_perf = True
    watch.meters['top5'].is_perf = True

    for epoch in range(99999):
        watch.tic(batch_size)
        time.sleep(0.1)
        watch.meters['top1'].update(1-np.exp(-epoch/10))
        watch.meters['top5'].update(1-np.exp(-epoch/5))
        watch.toc(loss=np.sin(epoch/10), lr=np.cos(epoch/20))

    watch.stop()

    pdb.set_trace()
