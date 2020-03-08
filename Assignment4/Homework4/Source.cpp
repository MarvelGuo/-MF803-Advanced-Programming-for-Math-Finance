#include <iostream>
#include <math.h>
#include <vector>
#include <tuple>

class ZeroCoupon {
public:
	int maturity;
	float YTM;
	float face_value;
	float coupon_price;
	float duration;
	float convexity;
	ZeroCoupon(int maturity_, float YTM_, float face_value_ = 100) {
		maturity = maturity_;
		YTM = YTM_;
		face_value = face_value_;
		coupon_price = price_cal(YTM);
		duration = duration_cal();
		convexity = convexity_cal2();
	};


	float price_cal(float YTM_) {
		return face_value / pow((1 + YTM_), maturity);
	}

	float duration_cal() {
		float price_up; float price_down;
		price_up = price_cal(YTM-0.01);
		price_down = price_cal(YTM + 0.01);

		return ((price_up-price_down) / 0.02/coupon_price);
	}

	float convexity_cal2() {
		float convexity = 0;
		float delta = 0.001;
		convexity = (price_cal(YTM + delta) + price_cal(YTM - delta) - 2 * price_cal(YTM)) / (delta * delta * price_cal(YTM));
		return  convexity;
	};

};

class Coupon : public ZeroCoupon {
public:
	float coupon_rate;
	float coupon_price;
	int pos;

	Coupon(int maturity_, float YTM_, float coupon_rate_ = 0, float face_value_ = 100, int pos_ = 0)
		:ZeroCoupon(maturity_, YTM_, face_value_) {
		coupon_rate = coupon_rate_;
		coupon_price = price_cal(YTM);
		duration = duration_cal();
		convexity = convexity_cal2();
		pos = pos_;
	};

	float price_cal(float YTM_) {
		float coupon_price_ = 0;
		//compute the coupon price by discount the cash flow
		for (int i = 1; i <= maturity; i++) {
			coupon_price_ += (face_value * coupon_rate) / pow((1 + YTM_), i);
		}

		coupon_price_ += face_value / pow((1 + YTM_), maturity);
		return coupon_price_;
	};

	float duration_cal() {
		float price_up; float price_down;
		price_up = price_cal(YTM - 0.005);
		price_down = price_cal(YTM + 0.005);

		return (price_up - price_down) / 0.01 / coupon_price;
	};

	float convexity_cal2() {
		float convexity = 0;
		float delta = 0.001;
		convexity = (price_cal(YTM + delta) + price_cal(YTM - delta) - 2 * price_cal(YTM)) / (delta * delta * price_cal(YTM));

		return  convexity;
	};
};

class Portfolio {
public:
	std::vector<Coupon> port;
	int port_size;
	double port_value;
	float port_duration;
	float port_convexity;

	Portfolio(std::vector<Coupon> port_) {
		port = port_;
		port_size = port_.size();
		port_value = port_value_cal(0);
		port_duration = port_duration_cal();
		port_convexity = port_convexity_cal();

	}

	double port_value_cal(float delta) {
		float value = 0;
		for (int i = 0; i < port_size; i++) {
			value += port.at(i).price_cal(port.at(i).YTM + delta) * port.at(i).pos;
		}
		return value;
	}


	float port_duration_cal() {
		float port_duration = 0;
		for (int i = 0; i < port_size; i++) {
			float price_up; float price_down; float price; float d;
			price = port.at(i).price_cal(port.at(i).YTM);
			price_up = port.at(i).price_cal(port.at(i).YTM - 0.001);
			price_down = port.at(i).price_cal(port.at(i).YTM + 0.001);
			d = (price_up - price_down) / (2 * 0.001 * price);
			port_duration += d * price / port_value * port.at(i).pos;
		}
		return port_duration;
	}


	float port_convexity_cal() {
		float port_conv = 0;
		for (int i = 0; i < port_size; i++) {
			float price_up; float price_down; float price; float conv;
			price = port.at(i).price_cal(port.at(i).YTM);
			price_up = port.at(i).price_cal(port.at(i).YTM - 0.001);
			price_down = port.at(i).price_cal(port.at(i).YTM + 0.001);
			conv = (price_up + price_down - 2*price) / ( 0.001 * 0.001 * price);
			port_conv += conv * price / port_value * port.at(i).pos;
		}
		return port_conv;
	}

};


int main() {
	std::vector<std::tuple<int, float>> data_list;
	data_list.push_back(std::make_tuple(1, 0.025));
	data_list.push_back(std::make_tuple(2, 0.026));
	data_list.push_back(std::make_tuple(3, 0.027));
	data_list.push_back(std::make_tuple(5, 0.03));
	data_list.push_back(std::make_tuple(10, 0.035));
	data_list.push_back(std::make_tuple(30, 0.04));

	std::vector<ZeroCoupon> vecCoupon;
	for (int i = 0; i <= 5; i++) {
		vecCoupon.push_back(ZeroCoupon(std::get<0>(data_list.at(i)), std::get<1>(data_list.at(i))));
	}

	// problem a
	std::cout << "Problem a:The corresponds zero coupon bonds' prices are following:" << std::endl;
	for (int i = 0; i <= 5; i++) {
		std::cout << vecCoupon.at(i).coupon_price << std::endl;
	}
	std::cout << "\n" << std::endl;

	// problem b
	std::cout << "Problem b:The durations of each bond are following:" << std::endl;
	for (int i = 0; i <= 5; i++) {
		std::cout << vecCoupon.at(i).duration << std::endl;
	}
	std::cout << "\n" << std::endl;

	// problem c
	std::cout << "Problem c: The prices of coupon bonds that pay $100 at maturity at 3% annually are following:" << std::endl;
	std::vector<Coupon> vecCoupon2;

	for (int i = 0; i <= 5; i++) {
		vecCoupon2.push_back(Coupon(std::get<0>(data_list.at(i)), std::get<1>(data_list.at(i)), 0.03, 100, 0));
	}

	for (int i = 0; i <= 5; i++) {
		std::cout << vecCoupon2.at(i).coupon_price << std::endl;
	}
	std::cout << "\n" << std::endl;

	// problem d
	std::cout << "Problem d: The duration of coupon bonds that pay $100 at maturity at 3% annually are following:" << std::endl;
	for (int i = 0; i <= 5; i++) {
		std::cout << vecCoupon2.at(i).duration << std::endl;
	}
	std::cout << "\n" << std::endl;

	// problem e
	std::cout << "Problem e: \nThe convexity of zero-coupon bonds are following:" << std::endl;
	for (int i = 0; i <= 5; i++) {
		std::cout << vecCoupon.at(i).convexity << std::endl;
	}
	std::cout << "The convexity of coupon bonds that pay $100 at maturity at 3% annually are following:" << std::endl;
	for (int i = 0; i <= 5; i++) {
		std::cout << vecCoupon2.at(i).convexity << std::endl;
	}
	std::cout << "\n" << std::endl;

	// problem f
	std::vector<Coupon> PortVec;
	PortVec.push_back(Coupon(1, 0.025, 0, 100, 1));
	PortVec.push_back(Coupon(2, 0.026, 0, 100, -2));
	PortVec.push_back(Coupon(3, 0.027, 0, 100, 1));
	Portfolio Port(PortVec);
	std::cout << "Problem f: The initial value of the portfolio is:" << std::endl;
	std::cout << Port.port_value << std::endl;
	std::cout << "\n" << std::endl;

	// problem g
	std::cout << "Problem g: \nThe duration of the portfolio is:" << std::endl;
	std::cout << Port.port_duration << std::endl;
	std::cout << "The convexity of the portfolio is:" << std::endl;
	std::cout << Port.port_convexity << std::endl;
	std::cout << "\n" << std::endl;

	//problem h
	std::vector<Coupon> PortVec2;
	PortVec2.push_back(Coupon(1, 0.035, 0, 100, 1));
	PortVec2.push_back(Coupon(2, 0.036, 0, 100, -2));
	PortVec2.push_back(Coupon(3, 0.037, 0, 100, 1));
	Portfolio Port2(PortVec2);
	std::cout << "Problem h: The initial value of the portfolio is:" << std::endl;
	std::cout << Port2.port_value << std::endl;
	std::cout << "\n" << std::endl;

	//problem i
	std::vector<Coupon> PortVec3;
	PortVec3.push_back(Coupon(1, 0.015, 0, 100, 1));
	PortVec3.push_back(Coupon(2, 0.016, 0, 100, -2));
	PortVec3.push_back(Coupon(3, 0.017, 0, 100, 1));
	Portfolio Port3(PortVec3);
	std::cout << "Problem i: The initial value of the portfolio is following:" << std::endl;
	std::cout << Port3.port_value << std::endl;
	std::cout << "\n" << std::endl;

	//problem j
	std::cout << "Problem j: The 5-year cashflow following:" << std::endl;
	float cashVec[5];
	for (int i = 0; i < 5; i++) {
		cashVec[i] = 0.2 * 100 + 0.03 * 100;
		std::cout << cashVec[i] << std::endl;
	}
	std::cout << "\n" << std::endl;

	//problem k
	float amortize_price = 0;
	float disc[5];
	disc[0] = 0.025; disc[1] = 0.026; disc[2] = 0.027; disc[3] = 0.0285; disc[4] = 0.03;
	//compute the coupon price by discount the cash flow
	for (int i = 0; i < 5; i++) {
		amortize_price += cashVec[i] / pow((1 + disc[i]), i + 1);
	}
	std::cout << "Problem k: The 5-year amortizing bond price is following:" << std::endl;
	std::cout << amortize_price << std::endl;

	float delta = 0.001;
	float amortize_price_p = 0;
	//compute the coupon price by discount the cash flow
	for (int i = 0; i < 5; i++) {
		amortize_price_p += cashVec[i] / pow((1 + disc[i] + delta), i + 1);
	}
	float amortize_price_m = 0;
	//compute the coupon price by discount the cash flow
	for (int i = 0; i < 5; i++) {
		amortize_price_m += cashVec[i] / pow((1 + disc[i] - delta), i + 1);
	}
	float amortize_price_duration = (amortize_price_m - amortize_price_p) / (2 * delta * amortize_price);
	std::cout << "Problem k: The 5-year amortizing duration is following:" << std::endl;
	std::cout << amortize_price_duration << std::endl;

	return 0;
}