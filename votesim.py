#!/usr/bin/python
# -*- encoding: UTF-8 -*-
'''Voter Simulation
    
    This section should be a summary of important infomation to help the editor
    understand the purpose and/or operation of the included code.
    
    Module dependencies:
        sys
    List of classes: -none-
    List of functions:
        main
'''

# built-in modules
import sys
import traceback
import random as rand
import re
import math
import cmath
from pprint import pprint

# third-party modules
from matplotlib import pyplot as plt

# self package

#===============================================================================
def main(*args):
    print 'Initializing candidates and electorate'
    candidates = [
        Candidate('Mr. Republican', 1.0, -1.0),
        Candidate('Mr. Democrat', -1.0, 1.0),
        Candidate('Ms. Populist', -1.0, -1.0),
        Candidate('Mr. Libertarian', 1.0, 1.0)
    ]
    pprint(candidates)
    z = cmath.rect( 1.0, 2*math.pi*rand.random() )
    Z = (z, -1*z)
    electorate = []
    for i in range(1000):
        z = rand.choice(Z)
        z += cmath.rect( rand.gauss(0.0, 0.5),
                        2*math.pi*rand.random()
                      )
        electorate.append( Voter(list(candidates), (z.real, z.imag)) )
    # END for
    #electorate = []
    #for i in range(1000):
    #    z = cmath.rect(rand.gauss(0.0, 1.2), 2*math.pi*rand.random())
    #    electorate.append( Voter(list(candidates), (z.real, z.imag)) )
    ## END for
    #electorate = [Voter(list(candidates)) for i in range(10000)]
    
    # Find the average position
    z = 0j
    for v in electorate:
        z += complex(*v.pos)
    # END for
    print 'center of preference is at '+str(z/len(electorate))
    
    # First poll should reflect sincere voting preferences
    print 'Calling for the first round of voting'
    polling_results = survey_electorate(electorate)
    print polling_results
    #print survey_least_prefered(electorate)
    print 'IRV results'
    pprint( get_IRV_results(electorate) )
    # Now allow electorate to vote with knowledge of the polls
    print 'Allowing voters to re-adjust votes to equillibrium'
    for i in range(120):
        new_poll_results = survey_electorate(electorate, polling_results)
        if new_poll_results == polling_results:
            polling_results = new_poll_results
            break
        # END if
        polling_results = new_poll_results
    # END
    
    if i == 0:
        print 'Voters are unmoved by the polling results'
    else:
        print '{} Rounds of re-adjustment have produced...'.format(i)
        print polling_results
    # END if
    
    # Now check what a ranked-pairs election would produce
    print 'Ranked-pairs sincere voting would produce the following...'
    cpoll = get_rankedpairs_results(electorate, candidates)
    pprint( cpoll.get_results() )
    print ''
    
    # Reintroduce centrist candidate
    #new_candidates = polling_results.get_rankings()
    new_candidates = [ Candidate('Ms. Centrist', 0, 0),
                       Candidate('Ms. Blue'),
                       Candidate('Ms. Green'),
                       Candidate('Ms. Red')
                     ]
    for nc in new_candidates[:2]:
        print 'Adding {} produces...'.format(repr(nc))
        candidates.append(nc)
        for v in electorate:
            v.reset_known_candidates(candidates)
        print survey_electorate(electorate)
        #print survey_least_prefered(electorate)
        print 'IRV results'
        pprint( get_IRV_results(electorate) )
        print 'Tideman RP results'
        cpoll = get_rankedpairs_results(electorate, candidates)
        #print cpoll
        pprint( cpoll.get_results() )
        print ''
    # END for
    
    graph_electorate(electorate, candidates)
# END main

#===============================================================================
class Candidate(object):
    '''Political Candidate Class
    '''
    
    def __init__(self, name, *xy):
        self.name = name
        if len(xy) >= 2:
            self.pos = tuple(xy[:2])
        else:
            self.pos = ( rand.gauss(0.0, 0.5), rand.gauss(0.0, 0.5))
        # END if
    # END __init__
    
    def __str__(self): return self.name
    def __repr__(self):
        return '{0} ({1[0]:0.2f},{1[1]:0.2f})'.format(self.name, self.pos)
    
# END Candidate

#===============================================================================
class Voter(object):
    '''Voter Class
    '''
    
    def __init__(self, known_candidates, pos=None):
        if len(known_candidates) == 0:
            raise ValueError(
                'Attempted to initialize a voter without any know candidates'
            )
        # END if
        
        if pos is None:
            self.pos = (2*rand.random()-1, 2*rand.random()-1)
        else:
            self.pos = pos
        # END if
        self.prefs = self.calc_prefs(known_candidates)
        #self.prefs = Voter.__make_rand_prefs(known_candidates)
        self.loyalty = math.sqrt( rand.random() )
    # END __init__
    
    @staticmethod
    def __make_rand_prefs(clist):
        out = []
        while len(clist) > 0:
            i = rand.randint(0,len(clist)-1)
            out.append( clist.pop(i) )
        # END while
        
        return out
    # END __make_rand_prefs
    
    def calc_prefs(self, poles):
        ''' Calculate the Voter's sincere preferences via their distance
        to each candidate in policy space
        '''
        distances = []
        for p in poles:
            distances.append( (p, pos_diff(p, self)) )
        distances.sort(key=lambda t: t[1])
        return [t[0] for t in distances]
    # END calc_prefs
    
    def get_vote(self, curr_poll=None):
        '''Get a vote for submission
        
        Voter will give a strategic vote based on prior knowledge, which is
        based on the curr_poll argument
        
        Arg:
            curr_poll (Poll): ranked list of candidates which the Voter
                              will perceive as the potential results
        '''
        if curr_poll is None: return self.prefs[0]
        
        p_fav = float(curr_poll[self.prefs[0]]) / curr_poll.N_votes
        if self.loyalty/len(curr_poll) < p_fav:
            # vote sincerely
            return self.prefs[0]
        else:
            # vote for a candidate with better chances to win than your favorite
            fav_rank = curr_poll.get_nth(self.prefs[0])
            alternatives = []
            for i in range(fav_rank):
                nth_alt = curr_poll.get_nth(i)
                alternatives.append( (nth_alt, self.prefs.index(nth_alt)) )
            # END for
            alternatives.sort(key=lambda t: t[1])
            
            return alternatives[0][0]
        # END if
    # END get_vote
    
    def get_condocert_vote(self, curr_poll=None):
        if curr_poll is None: return self.prefs[0]
        
        p_fav = float(curr_poll[self.prefs[0]]) / curr_poll.N_votes
        if self.loyalty/len(curr_poll) < p_fav:
            # vote sincerely
            return self.prefs[0]
        else:
            # vote for a candidate with better chances to win than your favorite
            fav_rank = curr_poll.get_nth(self.prefs[0])
            alternatives = []
            for i in range(fav_rank):
                nth_alt = curr_poll.get_nth(i)
                alternatives.append( (nth_alt, self.prefs.index(nth_alt)) )
            # END for
            alternatives.sort(key=lambda t: t[1])
            
            return alternatives[0][0]
        # END if
    # END get_vote
    
    def reset_known_candidates(self, new_candidates):
        self.prefs = self.calc_prefs(new_candidates)
    # END reset_known_candidates
# END Voter

#===============================================================================
def pos_diff(A, B):
    return math.sqrt(
        (A.pos[0]-B.pos[0])**2 + (A.pos[1]-B.pos[1])**2
    )
# END diff

#===============================================================================
class Poll(object):
    '''Polling results
    
    This class will resemble an ordered dictionary, where integer keys
    reference position and string keys reference dict-key.
    '''
    
    def __init__(self, tuplist=[]):
        if len(tuplist) > 1:
            tuplist.sort(key=lambda t: t[1], reverse=True)
        
        self.N_votes = 0
        self.__rankings = []
        self.__vote_totals = {}
        for t in self.__rankings:
            self.N_votes += t[1]
            self.__vote_totals[t[0]] = t[1]
            self.__rankings.append(t[0])
        # END for
        self.ordered = True
    # END __init__
    
    # Magic Methods
    #--------------
    def __contains__(self, candidate):
        return candidate in self.__vote_totals
    # END __contains__
    
    def __eq__(self, other):
        if len(self) != len(other):
            return False
        for c in other:
            if c not in self: return False
            if self[c] != other[c]: return False
        # END for
        
        return True
    # END __eq__
    
    def __getitem__(self, candidate):
        try:
            return self.__vote_totals[candidate]
        except KeyError:
            return 0
        # END try
    # END __getitem__
    
    def __iter__(self):
        if not self.ordered: self.__sort_poll()
        for i in range(len(self)):
            yield self.__rankings[i]
    # END __iter__
    
    def __len__(self): return len(self.__rankings)
    
    def __str__(self):
        out_str = '+' + 39*'-' + '+\n'
        for c in self:
            out_str += '| '
            out_str += '{0:17s}'.format(c)
            out_str += ' | '
            out_str += '{0:7d} ({1:6.3f}%)'.format(
                self[c], 100.0*float(self[c])/self.N_votes
            )
            out_str += ' |\n'
        # END for
        out_str += '+' + 39*'-' + '+\n'
        
        return out_str
    # END __str__
    
    # Poll-specific Methods
    #----------------------
    def add_vote(self, candidate):
        if candidate in self.__vote_totals:
            self.__vote_totals[candidate] += 1
            self.ordered = False
        else:
            self.__vote_totals[candidate] = 1
            self.__rankings.append(candidate)
        # END if
        self.N_votes += 1
    # END add_vote
    
    def get_rankings(self): return self.__rankings
    
    def get_nth(self, arg):
        '''Get Nth Candidate or Candidate Rank
        
        Rank index starts at 0 (i.e. the candidate with the most votes is the
        zeroth candidate).
        
        Arguments:
            arg (int|str): Argumenet type will determine the lookup direction
        Returns:
            (str|int)
        '''
        
        if self.ordered:
            if type(arg) is int:
                return self.__rankings[arg]
            elif type(arg) is Candidate:
                try:
                    return self.__rankings.index(arg)
                except ValueError:
                    return len(self)
                # END try
            else:
                raise TypeError(
                    'Input must be int or str, not '+type(arg).__name__
                )
            # END if
        else:
            self.__sort_poll()
            return self.get_nth(arg)
        # END if
    # END get_nth
    
    def __sort_poll(self):
        '''(Private) Sort Polling Results
        
        This method will be used to put the results back in order before any
        rankings are looked up.
        '''
        
        tuplist = []
        for candidate in self.__vote_totals:
            tuplist.append( (candidate, self.__vote_totals[candidate]) )
        tuplist.sort(key=lambda t: t[1], reverse=True)
        self.__rankings = []
        for t in tuplist:
            self.__rankings.append(t[0])
        
        self.ordered = True
    # END __sort_poll
# END Poll

#===============================================================================
def survey_electorate(electorate, last_poll=None):
    polling_results = Poll()
    for voter in electorate:
        polling_results.add_vote(voter.get_vote(last_poll))
    # END for
    
    return polling_results
# END survey_electorate

#===============================================================================
def survey_least_prefered(electorate, last_poll=None):
    polling_results = Poll()
    for voter in electorate:
        polling_results.add_vote(voter.prefs[-1])
    # END for
    
    return polling_results
# END survey_electorate

#===============================================================================
def get_IRV_results(electorate):
    '''
    Args:
        electorate (list): list of Voter objects
    '''
    removed_candidates = []
    ballots = []
    for voter in electorate:
        ballots.append(voter.prefs)
    # END for
    tally = {}
    for c in ballots[0]:
        tally[c.name] = []
    while tally.keys():
        # Distribute ballots to most-preferred remaining candidates
        for b in ballots:
            for i in range(len(b)):
                try:
                    tally[b[0].name].append(b)
                    break
                except KeyError:
                    pass
                # END try
            # END for
        # END for
        # check for a candidate with a majority of votes
        rankings = sorted( tally.keys(), key=lambda k: len(tally[k]),
                           reverse=True
                         )
        if tally[rankings[0]] > len(electorate)/2:
            # Declare the winner
            break
        # END if
        # No majority achieved, remove last-place candidate and repeat
        last_c = sorted(tally.keys(), key=lambda k: len(tally[k]))[0]
        removed_candidates.insert(0, rankings[-1])
        ballots = list( tally[rankings[-1]] )
        del tally[rankings[-1]]
    # END while
    
    return rankings + removed_candidates
# END get_IRV_results

#===============================================================================
def get_rankedpairs_results(electorate, known_candidates):
    poll = CondorcetPoll()
    for voter in electorate:
        poll.add_vote(voter.calc_prefs(known_candidates))
    poll.close_poll()
    return poll
# END get_rankedpairs_results

class CondorcetPoll(object):
    '''Polling results
    
    This class will resemble an ordered dictionary, where integer keys
    reference position and string keys reference dict-key.
    '''
    
    def __init__(self, tuplist=[]):
        self.N_votes = 0
        self.__rankings = []
        self.__pairs = {}
        self.poll_open = True
    # END __init__
    
    # Magic Methods
    #--------------
    def __getitem__(self, p): return self.__pairs[p]
    
    def __len__(self): return len(self.__pairs)
    
    def __str__(self):
        out_str = '+' + 50*'-' + '+\n'
        for p, vdiff in self._ordered_edges:
            out_str += '| '
            out_str += '{0:>4d} ({2:05.1%}): {1:31s}'.format(
                vdiff, p, (self.N_votes+vdiff)/(2.0*self.N_votes)
            )
            out_str += ' |\n'
        # END for
        out_str += '+' + 50*'-' + '+\n'
        
        return re.sub(r'>', ' > ', out_str)
    # END __str__
    
    # Poll-specific Methods
    #----------------------
    def add_vote(self, voter_prefs):
        for i in range(len(voter_prefs)-1):
            for j in range(i+1, len(voter_prefs)):
                # Voter supports candidate i over candidate i+1
                pair = voter_prefs[i].name + '>' + voter_prefs[j].name
                antipair = voter_prefs[j].name + '>' + voter_prefs[i].name
                if pair in self.__pairs:
                    self.__pairs[pair] += 1
                elif antipair in self.__pairs:
                    self.__pairs[antipair] -= 1
                else:
                    self.__pairs[pair] = 1
                # END if
            # END for
        # END for
        
        self.N_votes += 1
    # END add_vote
    
    def get_results(self):
        return self.__rankings
    # END get_results
    
    def close_poll(self):
        '''Lock poll and sort pairwise results
        '''
        
        tuplist = []
        # Invert any match-ups with a negative vote difference so that all
        # match-up strings read winner > loser
        for p in self.__pairs:
            vdiff = self.__pairs[p]
            if vdiff < 0:
                p = re.sub(r'([^>]+)>(.+)', r'\2>\1', p)
                vdiff *= -1
            # END if
            tuplist.append( (p,vdiff) )
        # END for
        tuplist.sort(key=lambda t: t[1], reverse=True)
        self._ordered_edges = tuplist
        
        self.poll_open = False
        
        if self.poll_open:
            raise RuntimeError(
                'asked for results of CondorcetPoll before it was closed'
            )
        # END if
        candidates = set()
        for p, _ in tuplist:
            c1, c2 = p.split('>')
            if c1 not in candidates:
                candidates.add(c1)
            if c2 not in candidates:
                candidates.add(c2)
        # END for
        G = DAG()
        for prefstr, _ in tuplist:
            new_edge = prefstr.split('>')
            G.add_edge(*new_edge)
            visited = set(new_edge[0])
            for v in G.iter_DFS(new_edge[0]):
                if v in visited:
                    G.dicard_edge(*new_edge)
                    break
                # END if
            # END for
        # END while
        self.__rankings = G.topo_sort()
    # END close_poll
# END CondorcetPoll

class DAG(object):
    '''
    '''
    
    def __init__(self, edges=[]):
        self._G = dict()
        for a, b in edges:
            self.add_edge(a, b)
        # END for
    # END __init__
    
    def __getitem__(self, k):
        return self._G[k]
    # END __getitem__
    
    def __str__(self):
        return str(self._G)
    # END __str__
    
    def add_edge(self, a, b):
        if a in self._G:
            self._G[a].add(b)
        else:
            self._G[a] = {b,}
        # END if
        if b not in self._G:
            self._G[b] = set()
    # END add_edge
    
    def discard_edge(self, a, b):
        if a in self._G:
            self._G[a].discard(b)
    # END discard_edge
    
    def iter_DFS(self, v_start, safety=True):
        n = 0
        stack = list(self._G[v_start])
        while stack:
            v = stack.pop(0)
            yield v
            n += 1
            if safety and 2*len(self._G.keys()) < n:
                break
            for neighbor in self._G[v]:
                stack.insert(0, neighbor)
        # END while
    # END iter_DFS
    
    def topo_sort(self):
        out = []
        G = dict(self._G)
        V = sorted(G.keys(), key=lambda v: len(G[v]))
        n_lim = len(V)**2 + 1
        n = 0
        while V:
            v = V.pop(0)
            if len(G[v]) == 0:
                out.insert(0, v)
                for v_other in V:
                    G[v_other].discard(v)
            else:
                V.append(v)
            # END if
            n += 1
            if n >= n_lim:
                pprint(G)
                pprint(V)
                pprint(out)
                raise RuntimeError('Infinite loop in DAG.topo_sort')
        # END while
        return out
    # END topo_sort
# END DAG

#===============================================================================
def graph_electorate(electorate, candidates):
    for voter in electorate:
        plt.plot([voter.pos[0]], [voter.pos[1]], 'bo', alpha=0.2)
    plt.plot([0.0], [0.0], 'o', color=(0.5,0.5,0.5))
    plt.plot([1.0], [1.0], 'o', color=(0, 1, 0))
    plt.plot([-1.0], [1.0], 'o', color=(0, 0, 0.5))
    plt.plot([1.0], [-1.0], 'o', color=(1, 0, 0))
    plt.plot([-1.0], [-1.0], 'o', color=(1, 0, 1))
    for c in candidates[-3:]:
        plt.plot(c.pos[0], c.pos[1], 'o')
    plt.plot([3,-3,-3,3], [3,3,-3,-3])
    plt.show()
    plt.close()
# END graph_electorate

#===============================================================================
if __name__ == '__main__':
    main()
    #try:
    #   main(*sys.argv[1:])
    #except Exception as err:
    #   exc_type, exc_value, exc_tb = sys.exc_info()
    #   bad_file, bad_line, func_name, text = traceback.extract_tb(exc_tb)[-1]
    #   print 'Error in {}'.format(bad_file)
    #   print '{} on {}: {}'.format(type(err).__name__, bad_line, err)
    #   print ''
    #finally:
    #   raw_input("press enter to exit")
    # END try
# END if