import numpy as np
import abc, six
try:
    import Queue as queue
except ImportError:
    import queue

@six.add_metaclass(abc.ABCMeta)
class DTALS_base():

    def __init__(self,tenpy,T,A):
        self.tenpy = tenpy
        self.T = T
        self.A = A
        self.R = A[0].shape[1]

    @abc.abstractmethod
    def _einstr_builder(self,M,s,ii):
        return

    @abc.abstractmethod
    def _solve(self,i,Regu,s):
        return

    def step(self,Regu):
        q = queue.Queue()
        for i in range(len(self.A)):
            q.put(i)
        s = [(list(range(len(self.A))),self.T)]
        while not q.empty():
            i = q.get()
            while i not in s[-1][0]:
                s.pop()
                assert(len(s) >= 1)
            while len(s[-1][0]) != 1:
                M = s[-1][1]
                idx = s[-1][0].index(i)
                ii = len(s[-1][0])-1
                if idx == len(s[-1][0])-1:
                    ii = len(s[-1][0])-2

                einstr = self._einstr_builder(M,s,ii)

                N = self.tenpy.einsum(einstr,M,self.A[ii])
                ss = s[-1][0][:]
                ss.remove(ii)
                s.append((ss,N))
            self.A[i] = self._solve(i,Regu,s)
        return self.A

@six.add_metaclass(abc.ABCMeta)
class DTLRALS_base():
    def __init__(self,tenpy,T,A,args,RHS=None):
        self.tenpy = tenpy
        self.T = T
        self.A = A
        self.R = A[0].shape[1]
        self.r = args.r
        self.RHS = RHS
        self.iterations = 0
        self.num_lowr_init_iter = args.num_lowr_init_iter

    @abc.abstractmethod
    def _einstr_builder(self,M,s,ii):
        return

    @abc.abstractmethod
    def _solve_DTLR(self,i,Regu,s):
        return

    @abc.abstractmethod
    def _solve_by_full_rank(self,i,Regu):
        return

    @abc.abstractmethod
    def _step_dt_subroutine(self,Regu):
        return

    def form_RHS(self):
        """Compute the right hand sides of all ALS subproblems.

        """
        self.RHS = []
        q = queue.Queue()
        for i in range(len(self.A)):
            q.put(i)
        s = [(list(range(len(self.A))),self.T)]
        while not q.empty():
            i = q.get()
            while i not in s[-1][0]:
                s.pop()
                assert(len(s) >= 1)
            while len(s[-1][0]) != 1:
                M = s[-1][1]
                idx = s[-1][0].index(i)
                ii = len(s[-1][0])-1
                if idx == len(s[-1][0])-1:
                    ii = len(s[-1][0])-2

                einstr = self._einstr_builder(M,s,ii)

                N = self.tenpy.einsum(einstr,M,self.A[ii])
                ss = s[-1][0][:]
                ss.remove(ii)
                s.append((ss,N))
            self.RHS.append(s[-1][1])
        return self.RHS

    def update_RHS(self,mode,U,VT):
        """Update the right hand side of all subproblems by using the low rank updates.

        Args:
            mode (int): the mode whose factor matrix got low rank update
            U (tenpy matrix): the first factor of the low rank update
            VT (tenpy matrix): the second factor of the low rank update
        """
        assert(self.RHS != None)
        q = queue.Queue()
        for i in range(len(self.A)):
            q.put(i)
        # Initialization steps
        s = [(list(range(len(self.A))),self.T)]
        einstr = self._einstr_builder_lr(self.T,s,mode)
        #print("Initializing einstr is ",einstr)
        N = self.tenpy.einsum(einstr,self.T,U)
        ss = s[-1][0][:]
        ss.remove(mode)
        s.append((ss,N))

        while not q.empty():
            i = q.get()
            if i == mode:
                continue
            while i not in s[-1][0]:
                s.pop()
                assert(len(s) >= 1)
            while len(s[-1][0]) != 1:
                M = s[-1][1]
                idx = s[-1][0].index(i)
                ii = s[-1][0][-1]
                if idx == len(s[-1][0])-1:
                    ii = s[-1][0][-2]

                einstr = self._einstr_builder_lr(M,s,ii)
                N = self.tenpy.einsum(einstr,M,self.A[ii])
                ss = s[-1][0][:]
                ss.remove(ii)
                s.append((ss,N))
            N = s[-1][1]
            self.RHS[i] += self.tenpy.einsum("iLR,LR->iR",N,VT)

    def step(self,Regu):
        """Perform one full sweep of low rank update method.

        Args:
            Regu (tenpy matrix): regularization factor.

        Returns:
            list: a list of factor matrices after one full sweep.

        """
        self.iterations += 1
        if self.iterations <= self.num_lowr_init_iter:
            return self._step_dt_subroutine(Regu)
        else:
            print("***** dimension tree low rank step *****")
            if self.RHS is None:
                self.form_RHS()
            for i in range(len(self.A)):
                [U,VT] = self._solve_DTLR(i,Regu)
                self.A[i] += self.tenpy.einsum("ij,jk->ik",U,VT)
                self.update_RHS(i,U,VT)
        return self.A

@six.add_metaclass(abc.ABCMeta)
class PPALS_base():
    """Pairwise perturbation optimizer

    Attributes:
        pp (bool): using pairwise perturbation or dimension tree to update.
        reinitialize_tree (bool): reinitialize the dimension tree or not.
        tol_restart_dt (float): tolerance for restarting dimention tree.
        tree (dictionary): store the PP dimention tree.
        order (int): order of the input tensor.
        dA (list): list of perturbation terms.

    References:
        Linjian Ma and Edgar Solomonik; Accelerating Alternating Least Squares for Tensor Decomposition
        by Pairwise Perturbation; arXiv:1811.10573 [math.NA], November 2018.
    """

    def __init__(self,tenpy,T,A,args):

        self.tenpy = tenpy
        self.T = T
        self.A = A

        self.pp = False
        self.reinitialize_tree = False
        self.tol_restart_dt = args.tol_restart_dt
        self.tree = { '0':(list(range(len(self.A))),self.T) }
        self.order = len(A)
        self.dA = []
        for i in range(self.order):
            self.dA.append(tenpy.zeros((self.A[i].shape[0],self.A[i].shape[1])))

    @abc.abstractmethod
    def _step_dt(self,Regu):
        return

    @abc.abstractmethod
    def _solve_PP(self,i,Regu,N):
        return

    @abc.abstractmethod
    def _get_einstr(self, nodeindex, parent_nodeindex, contract_index):
        """Build the Einstein string for the contraction.

        This function contract the tensor represented by the parent_nodeindex and
        the matrix represented by the contract_index and output the string.

        Args:
            nodeindex (numpy array): represents the contracted tensor.
            parent_nodeindex (numpy array): represents the contracting tensor.
            contract_index (int): index in self.A

        Returns:
            (string) A string used in self.tenpy.einsum

        """
        return

    def _get_nodename(self, nodeindex):
        """Based on the index, output the node name used for the key of self.tree.

        Args:
            nodeindex (numpy array): A numpy array containing the indexes that
                the input tensor is not contracted with.

        Returns:
            (string) A string correspoding to the input array.

        Example:
            When the input tensor has 4 dimensions:
            _get_nodename(np.array([1,2])) == 'bc'

        """
        if len(nodeindex) == self.order:
            return '0'
        return "".join([chr(ord('a')+j) for j in nodeindex])

    def _get_parentnode(self, nodeindex):
        """Get the parent node based on current node index

        Args:
            nodeindex (numpy array): represents the current tensor.

        Returns:
            parent_nodename (string): representing the key of parent node in self.tree
            parent_index (numpy array): the index of parent node
            contract_index (int): the index difference between the current and parent node

        """
        fulllist = np.array(range(self.order))

        comp_index = np.setdiff1d(fulllist, nodeindex)
        comp_parent_index = comp_index[:-1]

        contract_index = comp_index[-1]
        parent_index = np.setdiff1d(fulllist, comp_parent_index)
        parent_nodename = self._get_nodename(parent_index)

        return parent_nodename, parent_index, contract_index

    def _initialize_treenode(self, nodeindex):
        """Initialize one node in self.tree

        Args:
            nodeindex (numpy array): The target node index

        """
        nodename = self._get_nodename(nodeindex)
        parent_nodename, parent_nodeindex, contract_index = self._get_parentnode(nodeindex)
        einstr = self._get_einstr(nodeindex,parent_nodeindex,contract_index)

        if not parent_nodename in self.tree:
            self._initialize_treenode(parent_nodeindex)

        N = self.tenpy.einsum(einstr,self.tree[parent_nodename][1],self.A[contract_index])

        self.tree[nodename] = (nodeindex,N)

    def _initialize_tree(self):
        """Initialize self.tree
        """
        self.tree = { '0':(list(range(len(self.A))),self.T) }
        self.dA = []
        for i in range(self.order):
            self.dA.append(self.tenpy.zeros((self.A[i].shape[0],self.A[i].shape[1])))

        for ii in range(0, self.order):
            for jj in range(ii+1, self.order):
                self._initialize_treenode(np.array([ii,jj]))

        for ii in range(0, self.order):
            self._initialize_treenode(np.array([ii]))

    def _step_pp_subroutine(self,Regu):
        """Doing one step update based on pairwise perturbation

        Args:
            Regu (matrix): Regularization term

        Returns:
            A (list): list of decomposed matrices

        """
        print("***** pairwise perturbation step *****")
        for i in range(self.order):
            nodename = self._get_nodename(np.array([i]))
            N = self.tree[nodename][1]

            for j in range(i):
                parentname = self._get_nodename(np.array([j,i]))
                einstr = self._get_einstr(np.array([i]), np.array([j,i]), j)
                N += self.tenpy.einsum(einstr,self.tree[parentname][1],self.dA[j])
            for j in range(i+1, self.order):
                parentname = self._get_nodename(np.array([i,j]))
                einstr = self._get_einstr(np.array([i]), np.array([i,j]), j)
                N += self.tenpy.einsum(einstr,self.tree[parentname][1],self.dA[j])

            output = self._solve_PP(i,Regu,N)
            self.dA[i] = output - self.A[i]
            self.A[i] = output

        num_smallupdate = 0
        for i in range(self.order):
            if self.tenpy.sum(self.dA[i]**2)**.5 > self.tol_restart_dt:
                num_smallupdate += 1

        if num_smallupdate > 0:
            self.pp = False
            self.reinitialize_tree = False

        return self.A


    def _step_dt_subroutine(self,Regu):
        """Doing one step update based on dimension tree

        Args:
            Regu (matrix): Regularization term

        Returns:
            A (list): list of decomposed matrices

        """
        A_prev = self.A[:]
        self._step_dt(Regu)
        num_smallupdate = 0
        for i in range(self.order):
            self.dA[i] = self.A[i] - A_prev[i]
            if self.tenpy.sum(self.dA[i]**2)**.5 < self.tol_restart_dt:
                num_smallupdate += 1

        if num_smallupdate == self.order:
            self.pp = True
            self.reinitialize_tree = True
        return self.A

    def step(self,Regu):
        """Doing one step update in the optimizer

        Args:
            Regu (matrix): Regularization term

        Returns:
            A (list): list of decomposed matrices

        """
        if self.pp:
            if self.reinitialize_tree:
                self._initialize_tree()
                self.reinitialize_tree = False
            A = self._step_pp_subroutine(Regu)
        else:
            A = self._step_dt_subroutine(Regu)
        return A
