from __future__ import division
from madjax import wavefunctions
from madjax.madjax_patch import complex
from jax.numpy import where


def FFV2_0(F1,F2,V3,COUP):
    TMP0 = (F1[2]*(F2[4]*(V3[2]+V3[5])+F2[5]*(V3[3]+1j*(V3[4])))+F1[3]*(F2[4]*(V3[3]-1j*(V3[4]))+F2[5]*(V3[2]-V3[5])))
    vertex = COUP*-1j * TMP0
    return vertex



def FFV2_3(F1,F2,COUP,M3,W3):
    OM3 = 0.0
    OM3 = where(M3 != 0. , 1.0/M3**2, 0. )
    V3 = wavefunctions.WaveFunction(size=6)
    V3[0] = F1[0]+F2[0]
    V3[1] = F1[1]+F2[1]
    P3 = [-1.0*complex(V3[0]).real, -1.0*complex(V3[1]).real, -1.0*complex(V3[1]).imag, -1.0*complex(V3[0]).imag]
    TMP1 = (F1[2]*(F2[4]*(P3[0]+P3[3])+F2[5]*(P3[1]+1j*(P3[2])))+F1[3]*(F2[4]*(P3[1]-1j*(P3[2]))+F2[5]*(P3[0]-P3[3])))
    denom = COUP/(P3[0]**2-P3[1]**2-P3[2]**2-P3[3]**2 - M3 * (M3 -1j* W3))
    V3[2]= denom*(-1j)*(F1[2]*F2[4]+F1[3]*F2[5]-P3[0]*OM3*TMP1)
    V3[3]= denom*(-1j)*(-F1[2]*F2[5]-F1[3]*F2[4]-P3[1]*OM3*TMP1)
    V3[4]= denom*(-1j)*(-1j*(F1[2]*F2[5])+1j*(F1[3]*F2[4])-P3[2]*OM3*TMP1)
    V3[5]= denom*(-1j)*(-F1[2]*F2[4]-P3[3]*OM3*TMP1+F1[3]*F2[5])
    return V3


