# project-in-bit-academy
여기는 비트 아카데미의 응용 AI 프로그램 개발 강의에서 프로젝트를 진행하기 위해 개설된 깃허브 입니다

참가자: 한상근, 허순종, 최태훈, 박태민


함수 3개 중 1번째 함수에서 사용자가 지정한 이름으로 데이터베이스 및 컬렉션 생성  --> 지정된 이름으로 생성된 데이터베이스에 2, 3번째 함수 실행      


문제: 사용자가 지정한 이름은 def AAA(----)여기에 들어가는 부분이라 전역변수가 안먹힌다

따라서 2, 3번째 함수에 지정한 이름의 값을 넘겨줄 수가 없다   

class 상속을 사용해봤더니 넘겨줄 순 있지만 번호판 등 데이터를 입력할 때, 데이터베이스, 컬렉션 명 또한 계속 같이 입력해야 함,  
디폴트로 지정하려고 해도 사용자가 지정함에 따라 변동하는 값이기에 문자열로 특정할 수 없고 결국 전역변수가 안먹히는 동일한 문제
