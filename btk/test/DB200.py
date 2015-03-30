"""
Database example

this is an example how to customize the database module for a special
database. The dafault databse doesn't know about the right field names
of the utterance database so we have to overwrite the methods converting
database objects to Python objects and vice versa.
"""

from btk.dbase import Dbase, Utterance
import os


class DB200x(Dbase):
    def getUtterance(self, UID):
        """
        returns an utterance object with the specified utterance ID (SID) 
        """
        db = self._db.utt[UID]
        return Utterance(UID,
                         self,
                         db["SPEAKER"][0],
                         os.path.join(db["PATH"][0], db["ADC"][0]),
                         0, -1, db)

    def setUtterance(self, uid, utt):
        raise NotImplementedError, ""

db = DB200x("DB2005-spk", "DB2005-utt")

spk = db.getSpeaker("KPS_r020c")
for utt in spk.utt:
    print utt

# count the number of utterances of each speaker
for spk in db.spkIterator():
    i = 0
    for utt in spk.utt:  # better use i = len(spk.utt)
        i += 1           # this is only for demonstration
    print "%s\t%s" % (spk.sid, i)






