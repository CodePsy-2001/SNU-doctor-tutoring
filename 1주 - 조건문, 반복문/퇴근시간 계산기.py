import arrow

now = arrow.now()
leave = arrow.now().replace(hour=17, minute=30, second=0)

print("퇴근 시간:", leave - now)

print("퇴근 시간:", leave.humanize(granularity=['hour', 'minute', 'second'], locale='ko'))
