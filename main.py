##########################################################################
#DESC: Flask-based website created for inductive bias behavioral experiment
#      Inquires probabilistic confidence of users on different simulation datasets
#DATE: 01/08/2021
#NAME: Jong M. Shin
##########################################################################

# import packages
from flask import Flask, render_template, url_for, request, redirect, send_file
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine
from waitress import serve

from datetime import datetime
import numpy as np
import pandas as pd
import random
import string

import matplotlib
matplotlib.use('agg') 
# Currently, matplotlib is accessing the "tkagg" backend that connects to the GUI event loop 
# and that causes unexpected behaviour. The plain "agg" backend does not connect to the GUI at all.
# this allows plt.close()
# https://stackoverflow.com/questions/51188461/using-pyplot-close-crashes-the-flask-app

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle

import os
import io
import base64
import uuid
import pickle

# instantiation
app = Flask(__name__)
DATABASE_URL = os.environ.get('DATABASE_URL')
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
# app.config['DATABASE_URL'] = 'sqlite:///test.db' # for local database
db = SQLAlchemy(app)
# You can't use sqlite on Heroku. That's because it stores the db as a file, but the filesystem is ephemeral and not shared between dynos. 
# heroku run spins up a new dyno which only lasts for the duration of the command. 
# So it creates the db locally, and then immediately destroys everything including the new db.
# https://stackoverflow.com/questions/36224812/flask-migrate-doesnt-create-database-in-heroku

class generate:
   
    def generate_gaussian_parity(n, mean=np.array([-1, -1]), cov_scale=1, angle_params=None, k=1, acorn=None):
        if acorn is not None:
            np.random.seed(acorn)
            
        d = len(mean)
        lim = abs(mean[0])
        
        if mean[0] == -1 and mean[1] == -1:
            mean = mean + 1 / 2**k
        elif mean[0] == -2 and mean[1] == -2:
            mean = mean + 1
        
        mnt = np.random.multinomial(n, 1/(4**k) * np.ones(4**k))
        cumsum = np.cumsum(mnt)
        cumsum = np.concatenate(([0], cumsum))
        
        Y = np.zeros(n)
        X = np.zeros((n, d))
        
        for i in range(2**k):
            for j in range(2**k):
                temp = np.random.multivariate_normal(mean, cov_scale * np.eye(d), 
                                                    size=mnt[i*(2**k) + j])
                if abs(mean[0]) == 0.5:
                    temp[:, 0] += i*(1/2**(k-1))
                    temp[:, 1] += j*(1/2**(k-1))
                    
                elif abs(mean[0]) == 1:
                    temp[:, 0] += i*2
                    temp[:, 1] += j*2

                # screen out values outside the boundary
                idx_oob = np.where(abs(temp) > lim)
                
                for l in idx_oob:
                    
                    while True:
                        temp2 = np.random.multivariate_normal(mean, cov_scale * np.eye(d), 
                                                    size=1)

                        if (abs(temp2) < lim).all():
                            temp[l] = temp2
                            break
                
                X[cumsum[i*(2**k) + j]:cumsum[i*(2**k) + j + 1]] = temp
                
                if i % 2 == j % 2:
                    Y[cumsum[i*(2**k) + j]:cumsum[i*(2**k) + j + 1]] = 0
                else:
                    Y[cumsum[i*(2**k) + j]:cumsum[i*(2**k) + j + 1]] = 1
                    
        if d == 2:
            if angle_params is None:
                angle_params = np.random.uniform(0, 2*np.pi)
                
            R = generate.generate_2d_rotation(angle_params)
            X = X @ R
            
        else:
            raise ValueError('d=%i not implemented!'%(d))
        
        return X, Y.astype(int)

    def generate_2d_rotation(theta=0, acorn=None):
        if acorn is not None:
            np.random.seed(acorn)
        
        R = np.array([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)]
        ])
        
        return R

    def generate_uniform_XOR(b=1, N=100, r=False):

        boundary = np.random.multinomial(N, [1/4.]*4)
        bcum = np.cumsum(boundary)

        X = np.array([[0,0]])
        Y = np.zeros(N)
        Y[bcum[0]:bcum[2]] = 1
        ol = 0.0 # degree of overlap

        for i in range(2):
            for j in range(2):
                
                idx = 2*i+j

                if i == 1:
                    tempX = np.random.uniform(ol,-b,boundary[idx])
                else: 
                    tempX = np.random.uniform(-ol,b,boundary[idx])

                if j == 1:
                    tempY = np.random.uniform(ol,-b,boundary[idx])
                else:
                    tempY = np.random.uniform(-ol,b,boundary[idx])

                X = np.concatenate((X, np.c_[tempX, tempY]))

        if r:
            R = generate.generate_2d_rotation(np.pi/4)
            X = X @ R
        
        return X[1:], Y.astype(int)

    def generate_spirals(N, K=2, noise = 0.5, acorn = None, density=0.01, rng=1):

        #N number of poinst per class
        #K number of classes
        X, Y = [], []

        size = int(N/K)*rng # equal number of points per feature

        if K == 2:
            turns = 2
        
        # mvt = np.random.multinomial(N, 1/K * np.ones(K))
        
        if K == 2:
            # r = np.random.uniform(0, rng, size=size) #switched to static sampling to prevent contraction
            r = np.linspace(0, rng, size)
            r = np.sort(r)
            t = np.linspace(0,  np.pi * 4 * rng * turns/K, size) + noise * np.random.normal(0, density, size)
            dx = r * np.cos(t)
            dy = r * np.sin(t)

            X.append(np.vstack([dx, dy]).T)
            X.append(np.vstack([-dx, -dy]).T)
            Y += [0] * size 
            Y += [1] * size

        return np.vstack(X), np.array(Y).astype(int)

    def true_Uxor(l=-2, r=2, h=0.01):

        def generate_mask(l=-2, r=2, h=0.01):

            x = np.arange(l,r,h)
            y = np.arange(l,r,h)
            x,y = np.meshgrid(x,y)
            sample = np.c_[x.ravel(),y.ravel()]

            return sample#[:,0], sample[:,1]

        X = generate_mask(l=l, r=r, h=h)

        z = np.zeros(len(X),dtype=float) + 0.5

        for i, loc in enumerate(X):
            X0 = loc[0]
            X1 = loc[1]
            
            if X0 > l and X0 < 0 and X1 < r and X1 > 0:
                z[i] = 0
            elif X0 > 0 and X0 < r and X1 < r and X1 > 0:
                z[i] = 1
            elif X0 > l and X0 < 0 and X1 < 0 and X1 > l:
                z[i] = 1
            elif X0 > 0 and X0 < r and X1 < 0 and X1 > l:
                z[i] = 0

        return X[:,0],X[:,1],z

    def true_xor(l=-2, r=2, h=0.01, rotate=False, sig=0.25):

        def generate_mask(l=-2, r=2, h=0.01):
    
            x = np.arange(l,r,h)
            y = np.arange(l,r,h)
            x,y = np.meshgrid(x,y)
            sample = np.c_[x.ravel(),y.ravel()]

            return sample#[:,0], sample[:,1]
    
        X = generate_mask(l=l, r=r, h=h)

        def pdf(x, rotate=False, sig=0.25):
    
            # Generates true XOR posterior
            if rotate:
                mu01 = np.array([-0.5,0])
                mu02 = np.array([0.5,0])
                mu11 = np.array([0,0.5])
                mu12 = np.array([0,-0.5])
            else:
                mu01 = np.array([-0.5,0.5])
                mu02 = np.array([0.5,-0.5])
                mu11 = np.array([0.5,0.5])
                mu12 = np.array([-0.5,-0.5])
            cov = sig * np.eye(2)
            inv_cov = np.linalg.inv(cov) 

            p0 = (
                np.exp(-(x - mu01)@inv_cov@(x-mu01).T) 
                + np.exp(-(x - mu02)@inv_cov@(x-mu02).T)
            )/(2*np.pi*np.sqrt(np.linalg.det(cov)))

            p1 = (
                np.exp(-(x - mu11)@inv_cov@(x-mu11).T) 
                + np.exp(-(x - mu12)@inv_cov@(x-mu12).T)
            )/(2*np.pi*np.sqrt(np.linalg.det(cov)))

            # return p0-p1
            return p1/(p0+p1)
        
        z = np.zeros(len(X),dtype=float)

        for ii,x in enumerate(X):
            # if np.any([x <= -1.0, x >= 1.0]): #or x.any() > 1
            #     # z[ii] = 0.5
            #     pass
            # else:
            z[ii] = 1-pdf(x, rotate=rotate, sig=sig)#)/np.sqrt(4)
            # z[ii] = 1-pdf(x, rotate=rotate, sig=sig)

        z = (z - min(z)) / (max(z) - min(z))

        return X[:,0], X[:,1], z

class CmtList(db.Model):
    __tablename__ = 'Cmtdb'
    
    id = db.Column(db.Integer, primary_key=True)
    user = db.Column(db.String(200), default=str(uuid.uuid4()))
    comment = db.Column(db.String(1000), default="NA")

    def __repr__(self):
        return '<Task %r>' % self.id

class Todo(db.Model):
    __tablename__ = 'testdb'

    id = db.Column(db.Integer, primary_key=True)
    user = db.Column(db.String(200), default=str(uuid.uuid4()))
    hit = db.Column(db.Integer, default=999)
    trial = db.Column(db.Integer, nullable=False, default=999)
    mtype = db.Column(db.Integer, nullable=False, default=999) #limit # of string
    est = db.Column(db.Float, default=999.)
    real = db.Column(db.Float, default=999.)
    score = db.Column(db.Integer, default=999.)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)
    x = db.Column(db.String, default="999")
    sampleN = db.Column(db.Integer, default=999)

    def __repr__(self):
        return '<Task %r>' % self.id

# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    response.cache_control.no_store = True
    response.cache_control.max_age = 0
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

@app.route('/', methods=['POST', 'GET'])
def main():
    
    userid = str(uuid.uuid4())
    trial = 0
    init_real = 2
    mtype = 0
    x = 0
    sampleN = 100 #sampleN changed after the range increase from r=2 to r=3

    try: t = int(request.form['t'])
    except: t = 0

    try:
        score = int(request.form['score'])
        return render_template('index.html', user=userid, trial=trial, real=init_real, mtype=mtype, X=x, sampleN=sampleN, t=t, score=score)
    except:
        return render_template('index.html', user=userid, trial=trial, real=init_real, mtype=mtype, X=x, sampleN=sampleN, t=t)
    

@app.route('/read')
def read():
    tasks = Todo.query.order_by(Todo.user).all()
    return render_template('error.html', tasks=tasks)

@app.route('/readbyID')
def readbyID():
    tasks = Todo.query.order_by(Todo.id.desc()).all()    
    return render_template('error.html', tasks=tasks[:1000])

@app.route('/readbyDate')
def readbyDate():
    tasks = Todo.query.order_by(Todo.date_created.desc()).all()    
    return render_template('error.html', tasks=tasks[:1000])

@app.route('/tutorial', methods=['POST', 'GET'])
def tutorial():
    try: stage = int(request.form['stage']) + 1
    except: stage = 0
        # return "ERROR [2]: staging issue"
    try:
        score = int(request.form['score'])
        return render_template('tutorial.html', stage=stage, score=score)
    except:
        return render_template('tutorial.html', stage=stage)

@app.route('/test', methods=['POST', 'GET'])
def plot_fig():
 
    if request.method == 'POST':
        # # manual setup on trials
        # try:
        #     if int(request.form['hit']) < 3 or int(request.form['hit']) > 200:
        #         return redirect('/')
        # except:
        #     pass
        
        ############### INITIALIZATION ###############
        admin = 0 # admin mode 1 = True
        n = int(request.form['sampleN'])
        userid = str(request.form['user'])
        trial = int(request.form['trial'])

        # toggle square/circle on test.html
        try: inside = int(request.form['inside'])
        except: inside = 0

        # requesting data => n-1 acquisition
        try: hit = int(request.form['hit'])
        except: hit = 100 #hit changed from 50 to 100 on 1/28/2021

        real = float(request.form['real'])
        mtype = int(request.form['mtype'])
        x = str(request.form['X'])
        sampleN = int(request.form['sampleN'])

        try: #work in progress
            picklist = [
                int(request.form('dset0')),
                int(request.form('dset1')),
                int(request.form('dset2')),
                int(request.form('dset3')),
                int(request.form('dset4'))
            ]
        except:
            picklist = [2,4]#[0,1,2,3,4]
            
        pick = np.random.choice(picklist)

        # control parameters
        h = 0.1
        rng = 3
        tip = 0.1 #expanding window to cover the edge points
        five = False #activates five panel view

        #catch trial score
        try: 
            c_score = int(request.form['score'])
            c_real = int(request.form['c_real'])
        except: 
            c_score = 0
            c_real = 999

        # catch trial indexing
        try: cidx = str(request.form['cidx'])
        except: cidx = str(np.random.multinomial(hit*0.9, [1/5]*5).cumsum().tolist())
        ############### END INITIALIZATION ###############

        patience = 0.2 # threshold range for correct answer

        try: # request and calculate the score
            c_est = float(request.form['est'])

            if c_est <= c_real+patience and c_est >= c_real-patience:
                c_score += 1  

        except: pass

        for char in '[] ':
            cidx = cidx.replace(char,'')
        cidx = cidx.split(',')
        cidx = [int(i) for i in cidx if i != '']

        # conditional to introduce catch trial
        if len(cidx) != 0 and cidx[0] == trial:

            r_est = float(request.form['est'])
            cidx.pop(0)

            length = np.sqrt(np.random.uniform(0, 1, n))
            angle  = np.pi * np.random.uniform(0, 2, n)

            testx = length * np.cos(angle)
            testy = length * np.sin(angle)
            testX = np.c_[testx,testy]
            testY = (testX[:,trial%2] < 0)*1 #getting a counter to switch between horizontal and vertial split

            tempX = np.linspace(-1.8,1.8,20)
            tempY = np.linspace(-1.8,1.8,20)

            while True: #ensures sampling within the unit circle
                chooseX = np.random.choice(tempX)
                chooseY = np.random.choice(tempY)
                if np.sqrt(chooseX**2+chooseY**2) <= 1:
                    blckX = np.array([chooseX,chooseY])
                    break

            # getting a correct label
            if blckX[trial%2] < 0: blckY = 0 #getting a counter to switch between horizontal and vertial split
            else: blckY = 1

            ################ TESTING VIEW - CATCH ################
            fig, ax = plt.subplots()

            ax.scatter(testY*testX[:,0], testY*testX[:,1], linewidth=1, facecolors='none', edgecolors='green', s=30)
            ax.scatter(abs(testY-1)*testX[:,0], abs(testY-1)*testX[:,1], linewidth=1, facecolors='none', edgecolors='purple', s=30)
            ax.scatter(blckX[0],blckX[1], linewidth=1, facecolors='black', s=100)

            ax.axvline(c=[1.0, 0.5, 0.25], lw=2, alpha=0.5)
            ax.axhline(c=[1.0, 0.5, 0.25], lw=2, alpha=0.5)
            ax.axis([-2,2,-2,2]);
            ax.set_xticks([])
            ax.set_yticks([])

            img = io.BytesIO()
            fig.savefig(img, format='png', bbox_inches='tight')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode() 

            # try: plt.close()
            # except: pass
                    
            for char in '[] ':
                cidx = str(cidx).replace(char,'')  

            if trial < hit+1:
            # pushing current data for following request (i.e. pushing N, requesting N-1 to remain current)
                return render_template('test.html', imagen={'imagen': plot_url}, user=userid, score=c_score,
                        hit=hit, trial=trial, real=real, mtype=pick, sampleN=n, X=x, inside=inside, admin=admin, r_est=r_est,
                        c_real=blckY, cidx=cidx)

        if pick == 0: #GAUSS XOR
            X, Y = generate.generate_gaussian_parity(n=n, cov_scale=0.1, angle_params=np.pi)
            tempX, tempY, tempC = generate.true_xor(l=-rng, r=rng, h=h, rotate=False, sig=0.25)
            
        elif pick == 1: #GAUSS R-XOR
            X, Y = generate.generate_gaussian_parity(n=n, cov_scale=0.1, angle_params=np.pi/4)
            tempX, tempY, tempC = generate.true_xor(l=-rng, r=rng, h=h, rotate=True, sig=0.25)

        elif pick == 2: #GAUSS S-XOR
            X, Y = generate.generate_gaussian_parity(n=n, cov_scale=0.01, angle_params=np.pi)
            tempX, tempY, tempC = generate.true_xor(l=-rng, r=rng, h=h, rotate=False, sig=0.1)

        elif pick == 3: #UNIFORM XOR
            X, Y = generate.generate_uniform_XOR(N=n)
            tempX, tempY, tempC = generate.true_Uxor(l=-rng, r=rng, h=h)

        elif pick == 4: #SPIRAL
            X, Y = generate.generate_spirals(n, 2, noise=1, rng=1, density=0.3) #noise increased from 1.0 to 1.8 on 01/28/2021
            # tempX, tempY, tempC = generate.true_xor(h=h, rotate=False, sig=0.25)

            with open('static/clf/spiral.pickle', 'rb') as f:
                tempX, tempY, tempC = pickle.load(f, encoding='bytes')

        else:
            return "Unknown Error. Please restart the webpage."

        Y_test = tempC.copy()
        X_test = np.c_[(tempX,tempY)]
        newaxis = np.c_[(X,Y)]
        newaxis1 = np.c_[(X_test,Y_test)]
        np.random.shuffle(newaxis1)

        ################ TESTING VIEW ################
        fig, ax = plt.subplots()

        X1 = newaxis[newaxis[:,2]==0]
        ax.scatter(x=X1[:,0], y=X1[:,1], linewidth=1, facecolors='none', edgecolors='green', s=80)
        X1 = newaxis[newaxis[:,2]==1]
        ax.scatter(x=X1[:,0], y=X1[:,1], linewidth=1, facecolors='none', edgecolors='purple', s=80)

        ax.scatter(x=newaxis1[0,0], y=newaxis1[0,1], linewidth=1, facecolors='black', s=100)
        
        # ax.scatter(x=newaxis1[np.where(abs(newaxis1[:,0]) <= 1)[0][0],0], 
        #             y=newaxis1[np.where(abs(newaxis1[:,1]) <= 1)[0][0],1]
        #             , linewidth=1, facecolors='red', s=100) #testing inner samples
        # ax.scatter(x=rng, y=rng, linewidth=1, facecolors='black', s=100) #testing edge samples
        ax.scatter(X_test[:,0],X_test[:,1],c=Y_test, cmap='PRGn_r', alpha=0.2) #true posterior
        ax.axvline(c=[1.0, 0.5, 0.25], lw=2, alpha=0.5)
        ax.axhline(c=[1.0, 0.5, 0.25], lw=2, alpha=0.5)
        # ax.set_title(str(X_test) + str(y_test))
        ax.axis([-rng-tip,rng+tip,-rng-tip,rng+tip]);
        ax.set_xticks([])
        ax.set_yticks([])
  
        img = io.BytesIO()
        fig.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        try: 
            plt.close()
            # del img, fig, ax
        except: pass

        if admin == 1:
        ################ ADMIN VIEW ################
            fs = 2  #figure scale
            s  = 23 #point size
            faxis = np.array([-1,1,-1,1]) * fs#np.multiply([-1,1,-1,1], fs) 
            ftick = np.array([-1,0,1]) * fs#np.multiply([-1,0,1], fs) 

            fig, (ax1, ax2) = plt.subplots(1,2, figsize=(3*2,3), constrained_layout=True)
            plt.suptitle("X = " + str(newaxis1[0,0:2].round(3).tolist()) + " | Y = " + str(newaxis1[0,2].round(3)))
            
            #spiral not implemented yet
            if pick == 4:
                # ax1.scatter(x=newaxis1[:,0], y=newaxis1[:,1], c=newaxis1[:,2], cmap='RdBu_r', s=s)
                ax1.add_patch(Rectangle(
                    (-2,-2), 4, 4, linewidth=2, edgecolor='k', fill=False, hatch='/'))
                plt.suptitle("X = " + str(newaxis1[0,0:2].round(3).tolist()) + " | Y = Not Implemented Yet")
            else:
                ax1.scatter(x=newaxis1[:,0], y=newaxis1[:,1], c=newaxis1[:,2], cmap='PRGn_r', s=s)
                ax1.scatter(x=newaxis1[0,0], y=newaxis1[0,1], linewidth=1, facecolors='black', s=s)
            
            ax1.axvline(c=[1.0, 0.5, 0.25], lw=2)
            ax1.axhline(c=[1.0, 0.5, 0.25], lw=2)
            ax1.set_title('Posterior map')
            ax1.axis(faxis);
            ax1.set_yticks(ftick)
            ax1.set_xticks(ftick)

            ax2.scatter(x=newaxis1[:,0], y=newaxis1[:,1], linewidth=0.3, facecolors='none', edgecolors='black', s=s)
            ax2.scatter(x=newaxis1[0,0], y=newaxis1[0,1], linewidth=0.3, facecolors='black', s=s)
            ax2.set_title('Grid map')
            ax2.axvline(c=[1.0, 0.5, 0.25], lw=2)
            ax2.axhline(c=[1.0, 0.5, 0.25], lw=2)
            ax2.axis(faxis);
            ax2.set_xticks(ftick)
            ax2.set_yticks([])

            img = io.BytesIO()
            fig.savefig(img, format='png', bbox_inches='tight')
            img.seek(0)
            plot_url_admin = base64.b64encode(img.getvalue()).decode()
        else:
            plot_url_admin = None
        
        ################ 5 PANEL VIEW ################
        if five:
            newaxis_tot = []

            #GAUSS XOR
            X, Y = generate.generate_gaussian_parity(n=n, cov_scale=0.1, angle_params=np.pi)
            newaxis_tot.append(np.c_[(X,Y)])
                
            #GAUSS R-XOR
            X, Y = generate.generate_gaussian_parity(n=n, cov_scale=0.1, angle_params=np.pi/4)
            newaxis_tot.append(np.c_[(X,Y)])

            #GAUSS S-XOR
            X, Y = generate.generate_gaussian_parity(n=n, cov_scale=0.01, angle_params=np.pi)
            newaxis_tot.append(np.c_[(X,Y)])

            #UNIFORM XOR
            X, Y = generate.generate_uniform_XOR(b=1, N=n)
            newaxis_tot.append(np.c_[(X,Y)])

            #SPIRAL
            X, Y = generate.generate_spirals(n, 2, noise = 2.5, rng=1)
            newaxis_tot.append(np.c_[(X,Y)])
            
            # plt.suptitle("5 Panel view")

            modlst = ['GAUSS XOR','GAUSS R-XOR','GAUSS S-XOR','UNIFORM XOR','SPIRAL']

            s=5

            for ii in range(2):

                fig, ax = plt.subplots(2,3, figsize=(2*3,2*2), constrained_layout=True)

                for idx, i in enumerate(newaxis_tot):
                    X1 = i[i[:,2]==0]
                    ax[idx//3][idx%3].scatter(x=X1[:,0], y=X1[:,1], linewidth=1, facecolors='none', edgecolors='green', s=s)
                    X1 = i[i[:,2]==1]
                    ax[idx//3][idx%3].scatter(x=X1[:,0], y=X1[:,1], linewidth=1, facecolors='none', edgecolors='purple', s=s)
                    ax[idx//3][idx%3].scatter(x=newaxis1[0,0], y=newaxis1[0,1], linewidth=1, facecolors='black', s=s)

                    if ii == 0:
                        ax[idx//3][idx%3].add_patch(Rectangle(
                            (-1,-1), 2, 2, linewidth=1, edgecolor='k', fill=False))
                    else:
                        ax[idx//3][idx%3].add_patch(Circle(
                            (0,0), radius=1, edgecolor='k', fill=False))

                    ax[idx//3][idx%3].axvline(c=[1.0, 0.5, 0.25], lw=2, alpha=0.3)
                    ax[idx//3][idx%3].axhline(c=[1.0, 0.5, 0.25], lw=2, alpha=0.3)
                    ax[idx//3][idx%3].set_title(modlst[idx])
                    ax[idx//3][idx%3].axis([-2,2,-2,2]);
                    ax[idx//3][idx%3].set_xticks([])
                    ax[idx//3][idx%3].set_yticks([])

                ax[1][2].axis('off')
        
                img = io.BytesIO()
                fig.savefig(img, format='png', bbox_inches='tight')
                img.seek(0)
                if ii == 0:
                    plot_url_5 = base64.b64encode(img.getvalue()).decode()
                else:
                    plot_url_5_cir = base64.b64encode(img.getvalue()).decode()
    
        else:
            plot_url_5 = None
            plot_url_5_cir = None

        next_real = newaxis1[0,2]# if pick != 4 else 999

        # storing in db
        try:            
            # registering only the est when catch is not triggered
            if c_real != 999: #storing est from the real trial AFTER catch is triggered
                temp_est = float(request.form['r_est'])
                c_real = 999 #reset catch
            else: 
                temp_est = float(request.form['est'])                

            new_task = Todo(user=userid, hit=hit, trial=trial, mtype=mtype, est=temp_est, real=real, score=c_score, x=x, sampleN=sampleN)
            db.session.add(new_task)
            db.session.commit()              
  
        except: pass #initialization pass

        trial += 1 #starts from 1

        for char in '[] ':
            cidx = str(cidx).replace(char,'')

        # try: plt.close()
        # except: pass

        if trial <= hit:
            # pushing current data for following request (i.e. pushing N, requesting N-1 to remain current)
            label = str(newaxis1[0,0:2].round(3).tolist()).replace(' ','')
            return render_template('test.html', imagen={'imagen': plot_url}, imagen_admin={'imagen_admin': plot_url_admin}, 
                                                        imagen_5={'imagen_5': plot_url_5}, imagen_5_cir={'imagen_5_cir': plot_url_5_cir}, user=userid, score=c_score,
                                                        hit=hit, trial=trial, real=next_real, mtype=pick, sampleN=n, X=label, inside=inside, admin=admin, 
                                                        c_real=c_real, cidx=cidx)
                                                        #dset0=picklist[0],dset1=picklist[1],dset2=picklist[2],dset3=picklist[3],dset4=picklist[4])
        else:
            return render_template('finished.html', user=userid, done=1) 

@app.route('/catch', methods=["post"])
def catch_trial():

    # initialize
    try:
        chit = int(request.form['chit'])
        ctrial = int(request.form['ctrial'])
    except:
        chit = 5
        ctrial = 0

    n = 200
    ctrial += 1

    length = np.sqrt(np.random.uniform(0, 1, n))
    angle  = np.pi * np.random.uniform(0, 2, n)

    testx = length * np.cos(angle)
    testy = length * np.sin(angle)
    testX = np.c_[testx,testy]
    testY = (testX[:,ctrial%2] < 0)*1 #getting a counter to switch between horizontal and vertial split

    tempX = np.linspace(-1.8,1.8,20)
    tempY = np.linspace(-1.8,1.8,20)

    while True: #ensures sampling within the unit circle
        chooseX = np.random.choice(tempX)
        chooseY = np.random.choice(tempY)
        if np.sqrt(chooseX**2+chooseY**2) <= 1:
            blckX = np.array([chooseX,chooseY])
            break

    # getting a correct label
    if blckX[ctrial%2] < 0: blckY = 0 #getting a counter to switch between horizontal and vertial split
    else: blckY = 1

    patience = 0.2 # threshold range for correct answer

    # request and calculate the score
    try: 
        score = int(request.form['score'])
        est = float(request.form['est'])
        real = int(request.form['real'])
        if est <= real+patience and est >= real-patience:
            score += 1  
    except: 
        score = 0

    ################ TESTING VIEW ################
    fig, ax = plt.subplots()

    ax.scatter(testY*testX[:,0], testY*testX[:,1], linewidth=1, facecolors='none', edgecolors='green', s=30)
    ax.scatter(abs(testY-1)*testX[:,0], abs(testY-1)*testX[:,1], linewidth=1, facecolors='none', edgecolors='purple', s=30)
    ax.scatter(blckX[0],blckX[1], linewidth=1, facecolors='black', s=100)

    ax.axvline(c=[1.0, 0.5, 0.25], lw=2)
    ax.axhline(c=[1.0, 0.5, 0.25], lw=2)
    ax.axis([-2,2,-2,2]);
    ax.set_xticks([])
    ax.set_yticks([])

    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()   

    if ctrial < chit+1:
        return render_template('catch.html', imagen={'imagen': plot_url}, real=blckY, chit=chit, ctrial=ctrial, score=score)
    else:
        stage = 10
        return render_template('tutorial.html', score=score, stage=stage)

@app.route('/comment', methods=["POST"])
def comment():
    if request.method == 'POST':
        userid = str(request.form['user'])
        comment = str(request.form['comment'])
        new_task = CmtList(user=userid, comment=comment)
        db.session.add(new_task)
        db.session.commit()  
        return render_template('finished.html', user=userid, done=2)
    else:
        return 'ERROR [42]'

@app.route('/download', methods=["post"])
def downloadFile():
    path = ["test.xlsx", "test.db"]
    # postgresql+psycopg2://scott:tiger@localhost/mydatabase
    engine = create_engine(DATABASE_URL).connect()
    # engine = create_engine('sqlite:///test.db', echo=True).connect() 

    dswitch = int(request.form['dswitch'])

    if dswitch == 0:
        with pd.ExcelWriter(path[dswitch], engine='openpyxl') as writer:    
            output = pd.read_sql_table('testdb', con=engine)
            output.to_excel(writer, index=False, sheet_name='DB')

            output = pd.read_sql_table('Cmtdb', con=engine)
            output.to_excel(writer, index=False, sheet_name='COMMENTS')

            writer.save()
    
    return send_file(path[dswitch], as_attachment=True)

@app.route('/empty')
def emptydb():
    db.session.query(Todo).delete()
    db.session.query(CmtList).delete()
    db.session.commit()
    return redirect('/read')

if __name__ == "__main__":
    app.run(debug=True)
    # app.debug = True
    port = int(os.environ.get('PORT', 33507))
    serve(app, host='0.0.0.0', port=port)

