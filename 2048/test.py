
class Time:
    def __init__(self, h, m):
        self.h = self.checkHour(h)
        self.m = self.checkMinutes(m)

    def checkHour(self, h):
        if h > 23 or h < 0:
            return 0
        return h

    def checkMinutes(self, m):
        if m > 59 or m < 0:
            return 0
        return m

    def padZero(self, x):
        if x < 10:
            return "0" + str(x)
        return str(x)

    def toString(self):
        return self.padZero(self.h) + ":" + self.padZero(self.m)

    def minutesFromMidnight(self):
        return self.m + 60 * self.h

    def addMinutes(self, m):
        hoursToAdd = m // 60
        self.h += hoursToAdd % 24
        minutesToAdd = m % 60

        possibleMinutesToAdd = 60 - self.m

        if possibleMinutesToAdd > minutesToAdd:
            self.m += minutesToAdd
        elif possibleMinutesToAdd == minutesToAdd:
            self.m = 0
            self.h += 1
        else:
            self.m = minutesToAdd - possibleMinutesToAdd
            self.h += 1



time = Time(10, 31)
time.addMinutes(500)
print(time.toString())