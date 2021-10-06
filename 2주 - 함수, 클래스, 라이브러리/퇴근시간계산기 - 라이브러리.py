# 눈여겨볼 점: 외부 라이브러리 사용의 강력함!
import arrow

now = arrow.now()
leave = arrow.now().replace(hour=17, minute=30, second=0)

print("퇴근 시간:", leave - now)

print("퇴근 시간:", leave.humanize(granularity=['hour', 'minute', 'second'], locale='ko'))
