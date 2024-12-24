import numpy as np

total_bins = 48
phi = [i*3.14/17 for i in range(5,13)]
theta = [i*3.14/13 for i in range(4,10)]

# total_bins = 100
# phi = [i*3.14/19 for i in range(5,15)]
# theta = [i*3.14/19 for i in range(5,15)]

# total_bins = 64
# phi = [i*3.14/25 for i in range(9,17)]
# theta = [i*3.14/17 for i in range(5,13)]

def eucli_to_phi(coord):
    x = coord[0]
    y = coord[1]
    z = coord[2]
    phi = np.arccos(y)
        
    if x==0 and z==0:
        theta = 1.57
    else :
        theta = np.arccos(x/np.sqrt(x**2+z**2))
    
    return phi, theta


def phi_to_eucli(angles):
    phi = angles[0]
    theta = angles[1]
    y = np.cos(phi)
    x=1
    if theta!=1.57:
        z = np.tan(theta)
    else:
        z = 1
        x = 0
    if z<0:
        x*=-1
        z*=-1
    xz_norm = np.sqrt(x**2+z**2)
    if xz_norm>0 :
        x /= xz_norm
        z /= xz_norm
    else:
        x = 0
        z = 0
    x*= np.sqrt(1-y**2)
    z*= np.sqrt(1-y**2)
    
    return np.array([x,y,z])

def get_bin_num(coord):

    r_phi, r_theta = eucli_to_phi(coord)

    def bin_num(phi_id, theta_id):
        return phi_id*len(theta)+theta_id

    phi_ind = -1
    theta_ind = -1

    dist = 4
    for i,d in enumerate(phi):
        if dist>abs(d-r_phi):
            dist = abs(d-r_phi)
            phi_ind = i

    dist = 4
    for i,d in enumerate(theta):
        if dist>abs(d-r_theta):
            dist = abs(d-r_theta)
            theta_ind = i

    return bin_num(phi_ind, theta_ind)

def get_config(bin_num):
    # print('******',bin_num)
    return phi_to_eucli((phi[bin_num//len(theta)], theta[bin_num%len(theta)]))

def get_best_neigh(bin_num,coords):

    bin_coord = phi_to_eucli((phi[bin_num//len(theta)],theta[bin_num%len(theta)]))
    maxi_dot = -5
    ans = -1
    for i, d in enumerate(coords):
        if np.dot(d,bin_coord) > maxi_dot:
            ans = i
            maxi_dot = np.dot(d,bin_coord)
    return ans

def get_best_l(bins, lights):
    all_combi = []
    for i,s in enumerate(bins):
        bin_coord = phi_to_eucli((phi[s//len(theta)],theta[s%len(theta)]))
        for j,l in enumerate(lights):
            all_combi.append([i,j,np.dot(bin_coord,l)])
    all_combi = sorted(all_combi, key=lambda x: x[2], reverse=True)
    ans = [-1 for i in range(len(bins))]
    for i in all_combi:
        if ans[i[0]]==-1:
            ans[i[0]] = i[1]
    return ans