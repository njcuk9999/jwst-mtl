import numpy as np

import matplotlib.pyplot as plt

def gelmanrubin(*chain, burnin, npt):
    "Estimating PSRF"
    M = len(chain)  # number of chains
    N = chain[0].shape[0] - burnin  # assuming all chains have the same size.
    npars = chain[0].shape[1]  # number of parameters
    pmean = np.zeros(shape=(M, npars))  # allocate array to hold mean calculations
    pvar = np.zeros(shape=(M, npars))  # allocate array to hold variance calculations

    for i in range(0, M):
        currentchain = chain[i]
        for j in range(0, npars):
            pmean[i, j] = np.mean(currentchain[burnin:, j])  # Generate means for each parameter in each chain
            pvar[i, j] = np.var(currentchain[burnin:, j])  # Generate variance for each parameter in each chain

    posteriormean = np.zeros(npars)  # allocate array for posterior means
    for j in range(0, npars):
        posteriormean[j] = np.mean(pmean[:, j])  # calculate posterior mean for each parameter

    # Calculate between chains variance
    B = np.zeros(npars)
    for j in range(0, npars):
        for i in range(0, M):
            B[j] += np.power((pmean[i, j] - posteriormean[j]), 2)
    B = B * N / (M - 1.0)

    # Calculate within chain variance
    W = np.zeros(npars)
    for j in range(0, npars):
        for i in range(0, M):
            W[j] += pvar[i, j]
    W = W / M

    # Calculate the pooled variance
    V = (N - 1) * W / N + (M + 1) * B / (M * N)

    dof = npt - 1  # degrees of freedom
    Rc = np.sqrt((dof + 3.0) / (dof + 1.0) * V / W)  # PSRF from Brooks and Gelman (1997)

    # Calculate Ru
    # qa=0.95
    # ru=np.sqrt((dof+3.0)/(dof+1.0)*((N-1.0)/N*W+(M+1.0)/M*qa))

    return Rc;


def demhmcmc(x, llx, loglikelihood, beta, buffer, corbeta):
    "A Metropolis-Hastings MCMC with Gibbs sampler"

    nbuffer = len(buffer[:, 0])
    rsamp = np.random.rand()  # draw a random number to decide which sampler to use

    if rsamp < 0.5:  # if rsamp is less than 0.5 use a Gibbs sampler

        xt = np.copy(x)  # make a copy of our current state to the trail state
        npars = len(x)  # number of parameters
        n = int(np.random.rand() * npars)  # random select a parameter to vary.

        xt[n] = xt[n] + np.random.normal(0.0, beta[n])  # Step 2: Generate trial state with Gibbs sampler

    else:  # use our deMCMC sampler

        n = -1  # tell the accept array that we used the deMCMC sampler
        i1 = int(np.random.rand() * nbuffer)
        i2 = int(np.random.rand() * nbuffer)
        vectorjump = buffer[i1, :] - buffer[i2, :]
        xt = x + vectorjump * corbeta

    llxt = loglikelihood(xt)  # Step 3 Compute log(p(x'|d))=log(p(x'))+log(p(d|x'))

    alpha = min(np.exp(llxt - llx), 1.0)  # Step 4 Compute the acceptance probability

    u = np.random.rand()  # Step 5 generate a random number

    if u <= alpha:  # Step 6, compare u and alpha
        xp1 = np.copy(xt)  # accept new trial
        llxp1 = np.copy(llxt)
        ac = [0, n]  # Set ac to mark acceptance
    else:
        xp1 = np.copy(x)  # reject new trial
        llxp1 = np.copy(llx)
        ac = [1, n]  # Set ac to mark rejectance

    xp1 = np.array(xp1)
    return xp1, llxp1, ac;


def betarescale(x, beta, niter, burnin, mcmcfunc, loglikelihood, imax=20):
    "Calculate rescaling of beta to improve acceptance rates"

    alow = 0.22  # alow, ahigh define the acceptance rate range we want
    ahigh = 0.28

    delta = 0.01  # parameter controling how fast corscale changes - from Gregory 2011.

    npars = len(x)  # Number of parameters
    acorsub = np.zeros(npars)
    nacor = np.zeros(npars)  # total number of accepted proposals
    nacorsub = np.zeros(npars)  # total number of accepted proposals immediately prior to rescaling
    npropp = np.zeros(npars)  # total number of proposals
    nproppsub = np.zeros(npars)  # total number of proposals immediately prior to rescaling
    acrate = np.zeros(npars)  # current acrate
    corscale = np.ones(npars)

    # inital run
    chain, accept = genchain(x, niter, beta, mcmcfunc, loglikelihood)  # Get a MC
    nchain = len(chain[:, 0])

    # calcalate initial values of npropp and nacor
    for i in range(burnin, nchain):
        j = accept[i, 1]  # get accept flag value
        npropp[j] += 1  # update total number of proposals
        nacor[j] += 1 - accept[i, 0]  # update total number of accepted proposals

    # update x
    xin = chain[niter, :]  # we can continue to make chains by feeding the current state back into genchain

    acrate = nacor / npropp  # inital acceptance rate

    afix = np.ones(npars)  # afix is an integer flag to indicate which beta entries need to be updated
    for i in range(0, npars):
        if (acrate[i] < ahigh) & (acrate[i] > alow):  # we strive for an acceptance rate between alow,ahigh
            afix[i] = 0  # afix=1 : update beta, afix=0 : do not update beta

    # We will iterate a maximum of imax times - avoid infinite loops
    icount = 0  # counter to track iterations
    while (np.sum(afix) > 0):
        icount += 1  # track number of iterations

        if icount > 1:
            npropp = np.copy(nproppsub)
            nacor = np.copy(nacorsub)
        nacorsub = np.zeros(npars)  # reset nacorsub counts for each loop
        nproppsub = np.zeros(npars)  # reset nproppsub counts for each loop

        # Make another chain starting with xin
        betain = beta * corscale  # New beta for Gibbs sampling
        chain, accept = genchain(xin, niter, betain, mcmcfunc, loglikelihood)  # Get a MC
        xin = chain[niter, :]  # Store current parameter state

        for i in range(burnin, nchain):  # scan through Markov-Chains and count number of states and acceptances
            j = accept[i, 1]
            # if acrate[j]>ahigh or acrate[j]<alow:
            npropp[j] += 1  # update total number of proposals
            nacor[j] += 1 - accept[i, 0]  # update total number of accepted proposals
            nproppsub[j] += 1  # Update current number of proposals
            nacorsub[j] += 1 - accept[i, 0]  # Update current number of accepted proposals

        for i in range(0, npars):  # calculate acceptance rates for each parameter that is to updated
            # calculate current acrates
            acrate[i] = nacorsub[i] / nproppsub[i]

            # calculate acorsub
            acorsub[i] = (nacor[i] - nacorsub[i]) / (npropp[i] - nproppsub[i])

            if afix[i] > 0:
                # calculate corscale
                corscale[i] = np.abs(
                    corscale[i] * np.power((acorsub[i] + delta) * 0.75 / (0.25 * (1.0 - acorsub[i] + delta)), 0.25))

        print('Current Acceptance: ', acrate)  # report acceptance rates
        for i in range(0, npars):  # check which parameters have achieved required acceptance rate
            if acrate[i] < ahigh and acrate[i] > alow:
                afix[i] = 0

        if (icount > imax):  # if too many iterations, then we give up and exit
            afix = np.zeros(npars)
            print("Too many iterations: icount > imax")

    print('Final Acceptance: ', acrate)  # report acceptance rates

    return corscale;


def calcacrate(accept, burnin):  # ,label):
    "Calculate Acceptance Rates"
    nchain = len(accept[:, 0])
    print('%s %.3f' % ('Global Acceptance Rate:', (nchain - burnin - np.sum(accept[burnin:, 0])) / (nchain - burnin)))

    for j in range(max(accept[burnin:, 1]) + 1):
        denprop = 0  # this is for deMCMC
        deacrate = 0  # this is for deMCMC

        nprop = 0  # number of proposals
        acrate = 0  # acceptance rate

        for i in range(burnin, nchain):  # scan through the chain.
            if accept[i, 1] == j:
                nprop = nprop + 1
                acrate = acrate + accept[i, 0]
            if accept[i, 1] == -1:
                denprop = denprop + 1
                deacrate = deacrate + accept[i, 0]

        # print('%s Acceptance Rate %.3f' % (label[j],(nprop-acrate)/nprop))
        print('%s Acceptance Rate %.3f' % (str(j), (nprop - acrate) / (nprop + 1)))

    # if we have deMCMC results, report the acceptance rate.
    if denprop > 0:
        print('%s Acceptance Rate %.3f' % ('deMCMC', (denprop - deacrate) / denprop))

    return;


def genchain(x, niter, beta, mcmcfunc, loglikelihood, buffer=[], corbeta=1, progress=False):
    """Generate Markov Chain
    x - starting model parameters

    All variables needed by mcmcfunc are passed

    returns: chain, accept
        chain - Markov-Chain dimensions [npars,iter]
        accept - tracking acceptance
    """
    chain = []  # Initialize list to hold chain values
    accept = []  # Track our acceptance rate
    chain.append(x)  # Step 1: start the chain
    accept.append((0, 0))  # track acceptance rates for each parameter
    llx = loglikelihood(x)  # pre-compute the log-likelihood for Step 3

    if progress == True:
        for i in range(0, niter):
            x, llx, ac = mcmcfunc(x, llx, loglikelihood, beta, buffer, corbeta)
            chain.append(x)
            accept.append(ac)
    else:
        for i in range(0, niter):
            x, llx, ac = mcmcfunc(x, llx, loglikelihood, beta, buffer, corbeta)
            chain.append(x)
            accept.append(ac)

    chain = np.array(chain)  # Convert list to array
    accept = np.array(accept)

    return chain, accept;


def mhgmcmc(x, llx, loglikelihood, beta, buffer=[], corbeta=1):
    """A Metropolis-Hastings MCMC with Gibbs sampler
    x - np.array : independent variable
    llx - real : previous value of logL
    loglikeihood : returns log-likelihood
    beta - Gibb's factor : characteristic step size

    buffer - used when we discuss deMCMC

    returns: xp1,llxp1,ac
      xpl - next state (new parameters)
      llxp1 - log-likelihood of new state
      ac - if trial state was accepted or rejected.
    """

    xt = np.copy(x)  # make a copy of our current state to the trail state
    npars = len(x)  # number of parameters
    n = int(np.random.rand() * npars)  # random select a parameter to vary.

    xt[n] += np.random.normal(0.0, beta[n])  # Step 2: Generate trial state with Gibbs sampler

    llxt = loglikelihood(xt)  # Step 3 Compute log(p(x'|d))=log(p(x'))+log(p(d|x'))

    alpha = min(np.exp(llxt - llx), 1.0)  # Step 4 Compute the acceptance probability

    u = np.random.rand()  # Step 5 generate a random number

    if u <= alpha:  # Step 6, compare u and alpha
        xp1 = np.copy(xt)  # accept new trial
        llxp1 = np.copy(llxt)
        ac = [0, n]  # Set ac to mark acceptance
    else:
        xp1 = np.copy(x)  # reject new trial
        llxp1 = np.copy(llx)
        ac = [1, n]  # Set ac to mark rejectance

    return xp1, llxp1, ac;  # return new state and log(p(x|d))


def plotchains(chain, burnin, label, filename):
    npars = chain.shape[1]
    fig, ax = plt.subplots(nrows=npars, figsize=(12, 1.5 * npars))
    for i in range(npars):
        # fig[i].subplot(npars, 1, i+1)
        ax[i].plot(chain[burnin:, i])  # ,c=colour[i])
        ax[i].tick_params(direction='in', length=10, width=2)
        ax[i].set_ylabel(label[i])
        if i + 1 < npars:
            ax[i].set_xticklabels([])

    #plt.show()
    plt.savefig(filename)