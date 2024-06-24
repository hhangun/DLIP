#include <iostream>

namespace proj_A
{
	class myNum {
		public:

			int val1;
			int val2;
			int val3;

			myNum(int x1, int x2, int x3)
			{
				val1 = x1;
				val2 = x2;
				val3 = x3;
			}
			int sum()
			{
				int out = val1 + val2 + val3;
				return out;
			}
			void print()
			{
				std::cout << "val1: " << val1 << std::endl;
				std::cout << "val2: " << val2 << std::endl;
				std::cout << "val3: " << val3 << std::endl;
				std::cout << "sum =  " << sum() << std::endl;
			}
	};

}

namespace proj_B
{
	class myNum {
	public:

		int val1;
		int val2;
		int val3;

		myNum(int x1, int x2, int x3)
		{
			val1 = x1;
			val2 = x2;
			val3 = x3;
		}
		int sum()
		{
			int out2 = val1 + val2 + val3;
			return out2;
		}
		void print()
		{
			std::cout << "val1: " << val1 << std::endl;
			std::cout << "val2: " << val2 << std::endl;
			std::cout << "val3: " << val3 << std::endl;
			std::cout << "sum =  " << sum() << std::endl;
		}
	};
}

using namespace proj_A;

void main(){


	myNum mynum1(1, 2, 3);
	proj_B::myNum mynum2(4, 5, 6);

	mynum1.print();
	mynum2.print();

	system("pause");
}