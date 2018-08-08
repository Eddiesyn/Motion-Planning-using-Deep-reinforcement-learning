#coding:utf-8
import pyttsx3

class speek(object):
  """ The pronunciation"""
  def __init__(self):
    # super(ClassName, self).__init__()
    # self.arg = arg
    pass
  def sayingFind(self):
    eng1 = pyttsx3.init()
    eng1.say(u'I find it, excited')
    eng1.runAndWait()

    # eng1.runAndWait()
  def sayingSearch(self):
    eng2 = pyttsx3.init()
    eng2.say(u'I am searching')
    eng2.runAndWait()

  def sayingReset(self):
    eng3 = pyttsx3.init()
    eng3.say(u'I reset myself')
    eng3.runAndWait()  

  def sayingBye(self):
    eng4 = pyttsx3.init()
    eng4.say(u'Too young too simple')
    eng4.runAndWait()

  def sayingNo(self):
    eng4 = pyttsx3.init()
    eng4.say(u'I am angry')
    eng4.runAndWait()  



if __name__ == '__main__':
  cls = speek()
  cls.sayingSearch()
  cls.sayingFind()
