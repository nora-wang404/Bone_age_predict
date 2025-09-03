import glob
import matplotlib.pyplot as plt

losses = []
mads = []
with open("./trainlog.txt","r" ) as f:
    lline = f.readlines()
    for ll in lline:
        if ll.find(", loss=")>0 and ll.find("rank_zero.py ")>0:
            los=  float(ll.split(", loss=")[-1].split(", v_num=")[0].strip())
            losses.append(los)
        if ll.find("mad:")>0 and ll.find(", rmse:")>0:
            mad=  float(ll.split("mad:")[-1].split(", rmse:")[0].strip())
            mads.append(mad)


plt.title('loss')
plt.plot(list(range(0,len(losses))), losses, color='red'  )
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig("./output/loss.png")
plt.close()

plt.title('mad')
plt.plot(list(range(0,len(mads))), mads, color='red'  )
plt.xlabel('epoch')
plt.ylabel('mad')
plt.savefig("./output/mad.png")

plt.close()




