'''
Copyright (c) 2010 Carnegie Mellon University and Intel Corporation
  Author: Dmitry Berenson <dberenso@cs.cmu.edu>

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor Carnegie Mellon University,
      nor the names of their contributors, may be used to endorse or
      promote products derived from this software without specific prior
      written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
  ARE DISCLAIMED. IN NO EVENT SHALL INTEL CORPORATION OR CARNEGIE MELLON
  UNIVERSITY BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
  OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
  ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

from numpy import sin, cos, dot, eye
from numpy.linalg import norm
import numpy as np


def rodrigues(r):
    """Matrix that rotates points so that points have new orientation.

    Parameters
    ----------
    r : numpy.ndarray
        new normal vector

    Edited from:
    https://github.com/robEllenberg/comps-plugins/blob/master/python/rodrigues.py

    Theory:
    https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula

    Last modified: Lucas Sawade, 2020.11.16 16.30.00 (lsawade@princeton.edu)

    """
    def S(n):
        Sn = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
        return Sn
    theta = norm(r)
    if theta > 1e-30:
        n = r/theta
        Sn = S(n)
        R = eye(3) + sin(theta)*Sn + (1-cos(theta))*dot(Sn, Sn)
    else:
        Sr = S(r)
        theta2 = theta**2
        R = eye(3) + (1-theta2/6.)*Sr + (.5-theta2/24.)*dot(Sr, Sr)
    return R
