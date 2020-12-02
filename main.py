##########################################################################
#DESC: Flask-based website created for induced bias behavioral experiment
#      Inquires confidence of users for five different datasets
#DATE: 11/17/2020
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

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import os
import io
import base64
import uuid

# instantiation
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
db = SQLAlchemy(app)

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

    def generate_spirals(N, K=5, noise = 0.5, acorn = None, density=0.3, rng = 1):

        #N number of poinst per class
        #K number of classes
        X = []
        Y = []

        size = int(N/K) # equal number of points per feature

        if acorn is not None:
            np.random.seed(acorn)
        
        if K == 2:
            turns = 2
        elif K==3:
            turns = 2.5
        elif K==5:
            turns = 3.5
        elif K==7:
            turns = 4.5
        elif K==1:
            turns = 1
        else:
            print ("sorry, can't currently surpport %s classes " %K)
            return
        
        mvt = np.random.multinomial(N, 1/K * np.ones(K))
        
        if K == 2:
            r = np.random.uniform(0, rng, size=size)
            r = np.sort(r)
            t = np.linspace(0,  np.pi * 4 * turns/K, size) + noise * np.random.normal(0, density, size)
            dx = r * np.cos(t)
            dy = r * np.sin(t)

            X.append(np.vstack([dx, dy]).T)
            X.append(np.vstack([-dx, -dy]).T)
            Y += [0] * size 
            Y += [1] * size
        else:    
            for j in range(1, K+1):
                r = np.linspace(0.01, rng, int(mvt[j-1]))
                t = np.linspace((j-1) * np.pi *4 *turns/K,  j* np.pi * 4* turns/K, int(mvt[j-1])) + noise * np.random.normal(0, density, int(mvt[j-1]))
                dx = r * np.cos(t)
                dy = r * np.sin(t)

                dd = np.vstack([dx, dy]).T        
                X.append(dd)
                #label
                Y += [j-1] * int(mvt[j-1])
        return np.vstack(X), np.array(Y).astype(int)

    def true_Uxor(l=-2, r=2, h=0.01):

        def generate_mask(l=-2, r=2, h=0.01):

            x = np.arange(l,r,h)
            y = np.arange(l,r,h)
            x,y = np.meshgrid(x,y)
            sample = np.c_[x.ravel(),y.ravel()]

            return sample#[:,0], sample[:,1]

        X = generate_mask(l=l, r=r, h=h)

        # l=-1
        # r=1

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

class Todo(db.Model):

    __tablename__ = 'testdb'

    id = db.Column(db.Integer, primary_key=True)
    user = db.Column(db.String(200), default=str(uuid.uuid4()))
    hit = db.Column(db.Integer, default=999)
    trial = db.Column(db.Integer, nullable=False, default=999)
    mtype = db.Column(db.Integer, nullable=False, default=999) #limit # of string
    est = db.Column(db.Float, default=999.)
    real = db.Column(db.Float, default=999.)
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
    sampleN = 300

    try:
        t = int(request.form['t'])
    except:
        t = 0

    return render_template('index.html', user=userid, trial=trial, real=init_real, mtype=mtype, X=x, sampleN=sampleN, t=t)
    

@app.route('/read')
def read():
    tasks = Todo.query.order_by(Todo.user).all()

    return render_template('error.html', tasks=tasks[-1000:])

@app.route('/readbyID')
def readbyID():
    tasks = Todo.query.order_by(Todo.id).all()
    
    return render_template('error.html', tasks=tasks[-1000:])

@app.route('/readbyDate')
def readbyDate():
    tasks = Todo.query.order_by(Todo.date_created.desc()).all()
    
    return render_template('error.html', tasks=tasks[-1000:])

@app.route('/tutorial', methods=['POST', 'GET'])
def tutorial():
    try:
        stage = int(request.form['stage']) + 1
    except:
        stage = 0
        # return "ERROR [2]: staging issue"

    return render_template('tutorial.html', stage=stage)

@app.route('/test', methods=['POST', 'GET'])
def plot_fig():
 
    if request.method == 'POST':

        if int(request.form['hit']) < 3 or int(request.form['hit']) > 200:
            return redirect('/')

        n = int(request.form['sampleN'])

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

        h = 0.1
        five = False #activates backend for five panel view

        if pick == 0: #GAUSS XOR
            X, Y = generate.generate_gaussian_parity(n=n, cov_scale=0.1, angle_params=np.pi)
            tempX, tempY, tempC = generate.true_xor(h=h, rotate=False, sig=0.25)
            
        elif pick == 1: #GAUSS R-XOR
            X, Y = generate.generate_gaussian_parity(n=n, cov_scale=0.1, angle_params=np.pi/4)
            tempX, tempY, tempC = generate.true_xor(h=h, rotate=True, sig=0.25)

        elif pick == 2: #GAUSS S-XOR
            X, Y = generate.generate_gaussian_parity(n=n, cov_scale=0.01, angle_params=np.pi)
            tempX, tempY, tempC = generate.true_xor(h=h, rotate=False, sig=0.1)

        elif pick == 3: #UNIFORM XOR
            X, Y = generate.generate_uniform_XOR(N=n)
            tempX, tempY, tempC = generate.true_Uxor(h=h)

        elif pick == 4: #SPIRAL
            X, Y = generate.generate_spirals(n, 2, noise = 2.5, rng=2)
            tempX, tempY, tempC = generate.true_xor(h=h, rotate=False, sig=0.25)

            newX = np.array([[0,0]])
            newC = []

            for ii, i in enumerate(X):
                if np.all(np.abs(i) <= 1):
                    newX = np.concatenate((newX,[i]), axis=0)
                    newC.append(Y[ii])

            X = newX[1:]
            Y = np.array(newC)

        else:
            return "Unknown Error. Please restart the webpage."

        Y_test = tempC.copy()
        X_test = np.c_[(tempX,tempY)]
        newaxis = np.c_[(X,Y)]
        newaxis1 = np.c_[(X_test,Y_test)]
        np.random.shuffle(newaxis1)

        ######## TESTING VIEW ########
        fig, ax = plt.subplots()

        X1 = newaxis[newaxis[:,2]==0]
        ax.scatter(x=X1[:,0], y=X1[:,1], linewidth=1, facecolors='none', edgecolors='purple', s=30)
        X1 = newaxis[newaxis[:,2]==1]
        ax.scatter(x=X1[:,0], y=X1[:,1], linewidth=1, facecolors='none', edgecolors='green', s=30)

        ax.scatter(x=newaxis1[0,0], y=newaxis1[0,1], linewidth=1, facecolors='black', s=30)
        ax.axvline(c=[1.0, 0.5, 0.25], lw=2)
        ax.axhline(c=[1.0, 0.5, 0.25], lw=2)
        # ax.set_title(str(X_test) + str(y_test))
        ax.axis([-2,2,-2,2]);
        ax.set_xticks([])
        ax.set_yticks([])
  
        img = io.BytesIO()
        fig.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        ######## ADMIN VIEW ########
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
            ax1.scatter(x=newaxis1[:,0], y=newaxis1[:,1], c=newaxis1[:,2], cmap='PRGn', s=s)
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
        
        if five:
            ######## 5 PANEL VIEW ########
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
                    ax[idx//3][idx%3].scatter(x=X1[:,0], y=X1[:,1], linewidth=1, facecolors='none', edgecolors='purple', s=s)
                    X1 = i[i[:,2]==1]
                    ax[idx//3][idx%3].scatter(x=X1[:,0], y=X1[:,1], linewidth=1, facecolors='none', edgecolors='green', s=s)
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

        # toggle square/circle on test.html
        try: inside = int(request.form['inside'])
        except: inside = 0

        # requesting data => n-1 acquisition
        hit = int(request.form['hit'])
        userid = str(request.form['user'])
        trial = int(request.form['trial'])
        real = request.form['real']
        next_real = newaxis1[0,2] if pick != 4 else 999
        mtype = int(request.form['mtype'])
        x = str(request.form['X'])
        sampleN = int(request.form['sampleN'])
        
        try:
            est = float(request.form['est'])
            new_task = Todo(user=userid, hit=hit, trial=trial, mtype=mtype, est=est, real=real, x=x, sampleN=sampleN)
            db.session.add(new_task)
            db.session.commit()                
        except:
            pass

        trial += 1

        if trial < hit+1:
            # pushing current data for following request (i.e. pushing N, requesting N-1 to remain current)
            label = str(newaxis1[0,0:2].round(3).tolist()).replace(' ','')
            return render_template('test.html', imagen={'imagen': plot_url}, imagen_admin={'imagen_admin': plot_url_admin}, 
                                                        imagen_5={'imagen_5': plot_url_5}, imagen_5_cir={'imagen_5_cir': plot_url_5_cir}, user=userid, 
                                                        hit=hit, trial=trial, real=next_real, mtype=pick, sampleN=n, X=label, inside=inside)
                                                        #dset0=picklist[0],dset1=picklist[1],dset2=picklist[2],dset3=picklist[3],dset4=picklist[4])
        else:
            return render_template('finished.html', user=userid) 


@app.route('/download', methods=["post"])
def downloadFile():
    path = ["test.xlsx", "test.db"]
    engine = create_engine('sqlite:///test.db', echo=True).connect() 
    output = pd.read_sql_table('testdb', con=engine)
    output.to_excel('test.xlsx', index=False)
    
    return send_file(path[0], as_attachment=True)

@app.route('/empty')
def emptydb():
    db.session.query(Todo).delete()
    db.session.commit()
    return redirect('/read')

if __name__ == "__main__":
    app.run(debug=True)
    # app.debug = True
    # port = int(os.environ.get('PORT', 33507))
    # serve(app, host='0.0.0.0', port=port)

