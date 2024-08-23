import numpy as np
import matplotlib.pyplot as plt
import numpy.random as nr
from scipy.special import comb

blockLength = 1000; # Number of symbols per block
nBlocks = 10000; # Number of blocks
EbdB = np.arange(1.0,18.1,1.5); # Energy per bit in dB
Eb = 10**(EbdB/10); # Energy per bit Eb
No = 1; # Total noise power No
SNR = 2*Eb/No; # Signal-to-noise power ratio
SNRdB = 10*np.log10(Eb/No); # SNR values in dB
BER1 = np.zeros(len(EbdB)); # Bit error rate (BER) values
BERt1 = np.zeros(len(EbdB)); # Analytical values of BER from formula
BER2 = np.zeros(len(EbdB)); # Bit error rate (BER) values
BERt2 = np.zeros(len(EbdB)); # Analytical values of BER from formula
BER3= np.zeros(len(EbdB)); # Bit error rate (BER) values
BERt3 = np.zeros(len(EbdB)); # Analytical values of BER from formula

L = 1; # Number of antennas
for blk in range(nBlocks):    
    # Rayleigh fading channel coefficient with average power unity
    h = (nr.normal(0.0, 1.0,(L,1))+1j*nr.normal(0.0, 1.0,(L,1)))/np.sqrt(2);
    # Complex Gaussian noise with power No
    noise = nr.normal(0.0, np.sqrt(No/2), (L,blockLength))+1j*nr.normal(0.0, np.sqrt(No/2), (L,blockLength));
    BitsI = nr.randint(2,size=blockLength); # Bits for I channel
    BitsQ = nr.randint(2,size=blockLength); # Bits for Q channel
    Sym = (2*BitsI-1)+1j*(2*BitsQ-1); # Complex QPSK symbols

    for K in range(len(SNRdB)):
      
        TxSym=np.sqrt(Eb[K])*Sym;
        RxSym= h*TxSym+noise;
        
        MRCout =np.sum(np.conj(h)*RxSym,axis=0);
        DecBitsI=(np.real(MRCout)>0);
        DecBitsQ=(np.imag(MRCout)>0);
        
        BER1[K]=BER1[K]+np.sum(DecBitsI !=BitsI)+np.sum(DecBitsQ !=BitsQ)
        
BER1 = BER1/blockLength/nBlocks/2; # Evaluating BER from simulation 
BERt1 = comb(2*L-1, L)/2**L/SNR**L; # Evaluating BER from formula

L=2;
for blk in range(nBlocks):    
    # Rayleigh fading channel coefficient with average power unity
    h = (nr.normal(0.0, 1.0,(L,1))+1j*nr.normal(0.0, 1.0,(L,1)))/np.sqrt(2);
    # Complex Gaussian noise with power No
    noise = nr.normal(0.0, np.sqrt(No/2), (L,blockLength))+1j*nr.normal(0.0, np.sqrt(No/2), (L,blockLength));
    BitsI = nr.randint(2,size=blockLength); # Bits for I channel
    BitsQ = nr.randint(2,size=blockLength); # Bits for Q channel
    Sym = (2*BitsI-1)+1j*(2*BitsQ-1); # Complex QPSK symbols

    for K in range(len(SNRdB)):
        
        TxSym=np.sqrt(Eb[K])*Sym;
        RxSym= h*TxSym+noise;
        
        MRCout =np.sum(np.conj(h)*RxSym,axis=0);
        DecBitsI=(np.real(MRCout)>0);
        DecBitsQ=(np.imag(MRCout)>0);
        
        BER2[K]=BER2[K]+np.sum(DecBitsI !=BitsI)+np.sum(DecBitsQ !=BitsQ)
           
BER2 = BER2/blockLength/nBlocks/2; # Evaluating BER from simulation 
BERt2 = comb(2*L-1, L)/2**L/SNR**L; # Evaluating BER from formula

L=3;
for blk in range(nBlocks):    
    # Rayleigh fading channel coefficient with average power unity
    h = (nr.normal(0.0, 1.0,(L,1))+1j*nr.normal(0.0, 1.0,(L,1)))/np.sqrt(2);
    # Complex Gaussian noise with power No
    noise = nr.normal(0.0, np.sqrt(No/2), (L,blockLength))+1j*nr.normal(0.0, np.sqrt(No/2), (L,blockLength));
    BitsI = nr.randint(2,size=blockLength); # Bits for I channel
    BitsQ = nr.randint(2,size=blockLength); # Bits for Q channel
    Sym = (2*BitsI-1)+1j*(2*BitsQ-1); # Complex QPSK symbols

    for K in range(len(SNRdB)):
        
        TxSym=np.sqrt(Eb[K])*Sym;
        RxSym= h*TxSym+noise;
        
        MRCout =np.sum(np.conj(h)*RxSym,axis=0);
        DecBitsI=(np.real(MRCout)>0);
        DecBitsQ=(np.imag(MRCout)>0);
        
        BER3[K]=BER3[K]+np.sum(DecBitsI !=BitsI)+np.sum(DecBitsQ !=BitsQ)
            
BER3 = BER3/blockLength/nBlocks/2; # Evaluating BER from simulation 
BERt3 = comb(2*L-1, L)/2**L/SNR**L; # Evaluating BER from formula

# Plotting the bit error rate from Simulation and formula
plt.yscale('log');
plt.plot(SNRdB, BER1,'g-');
plt.plot(SNRdB, BERt1,'ro');
plt.plot(SNRdB, BER2,'b-');
plt.plot(SNRdB, BERt2,'yo');
plt.plot(SNRdB, BER3,'m-');
plt.plot(SNRdB, BERt3,'ko');
plt.grid(1,which='both')
plt.suptitle('BER for MRC(L=1,L=2,L=3)');
plt.legend(["BER L=1","BER theoretical L=1","BER L=2","BER theoretical L=2","BER L=3","BER theoretical L=3"],loc = "lower left") 

#plt.legend(["Simulation", "Theory"], loc ="lower left");
plt.xlabel('SNR (dB)')
plt.ylabel('BER') 
