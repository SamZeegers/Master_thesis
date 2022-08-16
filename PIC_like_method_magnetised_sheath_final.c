#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_math.h>
#include <time.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <omp.h>
#include <unistd.h>

//#include "nrutil.h"
//Defining some global variables that are used throughout the code the values are set in the main() function
double m_i; //ion mass
double e; //elementary charge
double Z; //effective charge an integer bigger than zero
double dx; //step size in space
int n; //number of grid points in space
double dv, dtheta;//, dlambda; //step size in velocity space
//double phi_0; //value of the potential at the wall
int p; //order of the finite element basis function
int steps_per_cell; //number of steps a trajectory takes in each grid cell.
double v_max, lambda_max, theta_max; //maximum velocity considered
double alpha;
double B;
double delta;
double kappa;
double Omega;
double T_e;
int g_flag;
double dg;
#define MAXIT 100
//FILE *fout_2;
//FILE *fout_3;
double normalization_factor;
#pragma omp threadprivate(dx, dtheta)
//double L;
/*
int fast_floor(float f){
	int i;
	
	__asm__ __volatile__(
	"	flds	%1\n"
	"	fistpl 	%0\n;"
	: "=m" (i) : "m" (f));
	//printf("%d\n", i);
	return i;
}
*/

void linear_element(double *x_nodes, int i_elm, double x, double *h, double *h_x, double length, int squared_flag){
	//This function calculates the finite function value of the finite elements that exist in the element i_elm
	//in the point x. h and h_x are pointers that will store the function values of the basis functions and its
	//derivative respectively. Note that in an element p+1 basis functions exist and thus h and h_x are p+1 in 
	//length. Length is a double that stores the grid size of the grid considered. x_nodes in a pointer that 
	//contains the x values of the specific grid points. 
	int node_1, node_2;
	double x1, x2;

	node_1 = i_elm;
	node_2 = i_elm+1;
	x1 = x_nodes[node_1]; //left boundary position
	//printf("%f\n", x);
	//printf("%f\n", x1); 
	x2 = x_nodes[node_2]; //position of the right boundary
	if (squared_flag == 1){
		x1 *= x1;
		x2 *= x2;
	}
	//printf("%f\n", x2);
	length = x2 - x1; //the grid size of this specific element
	
	//As for now 2 options are implemented linear or quadratic basis functions the quadratic basis functions
	//are defined below and are used when p(the order of the basis function) = 2.
	if(p==2){
		//the definitions of h[0], h[1] and h[2] are derived analytically h[0] always stores the basis function
		//in the element that is zero at x2, while h[1] always stores the basis function that is not
		//zero at x1 or x2. Finally h[2] stores the basis function which is zero at x1. Note that for quadratic
		//basis fucntions also the derivatives are zero when the basis function itself is zero. 
		//double length;
		double length_2;
		double length_2_prev;
		double x_1;
		double x_1_prev;

		if(i_elm == 0){
			h[0] = pow((x_nodes[1] - x),2.0)/(pow((x_nodes[1] - x_nodes[0]), 2.0));
			h_x[0] = -2.0*(x_nodes[1] - x)/(pow((x_nodes[1] - x_nodes[0]),2.0));
			h[2] = pow((x-x_nodes[0]), 2.0)/((x_nodes[2]-x_nodes[0])*(x_nodes[1] - x_nodes[0]));
			h_x[2] = 2.0*(x - x_nodes[0])/((x_nodes[2]-x_nodes[0])*(x_nodes[1] - x_nodes[0]));
			h[1] = 1.0 - h[0] - h[2]; 
			h_x[1] = -h_x[0] - h_x[2];
		}
		else if(i_elm == n-2){
			h[2] = pow((x - x_nodes[n-2]), 2.0)/(pow((x_nodes[n-1] - x_nodes[n-2]), 2.0));
			h_x[2] = 2.0*(x - x_nodes[n-2])/(pow((x_nodes[n-1] - x_nodes[n-2]), 2.0));
			h[0] = (1.0 - (x - x_nodes[n-3])/(x_nodes[n-1] - x_nodes[n-3]))*(1.0 - (x - x_nodes[n-2])/(x_nodes[n-1] - x_nodes[n-2]));
			h_x[0] = -1.0/(x_nodes[n-1] - x_nodes[n-3])*(1.0 - (x - x_nodes[n-2])/(x_nodes[n-1] - x_nodes[n-2])) - 1.0/(x_nodes[n-1] - x_nodes[n-2])*(1.0 - (x - x_nodes[n-3])/(x_nodes[n-1] - x_nodes[n-3]));
			h[1] = 1.0 - h[2] - h[0];
			h_x[1] = -h_x[2] - h_x[0];
		}
		else{
			length = x_nodes[i_elm+1] - x_nodes[i_elm];
			length_2 = x_nodes[i_elm+2] - x_nodes[i_elm];
			length_2_prev = x_nodes[i_elm+1] - x_nodes[i_elm-1];

			x_1 = x_nodes[i_elm];
			x_1_prev = x_nodes[i_elm - 1];
			h[2] = pow((x - x_1), 2.0)/(length*length_2);
			h_x[2] = 2.0*(x - x_1)/(length*length_2);
			h[1] = (x-x_1_prev)/(length_2_prev)*(1.0 - (x-x_1)/(length)) + (x-x_1)/(length)*(1.0 - (x-x_1)/length_2);
			h_x[1] = 1.0/length_2_prev + 1.0/length - (2.0*x - x_1_prev - x_1)/(length_2_prev*length) - (2.0*x - 2.0*x_1)/(length_2*length);
			h[0] = (1.0-(x-x_1_prev)/(length_2_prev))*(1.0- (x-x_1)/(length));
			h_x[0] = -1.0/(length_2_prev)*(1.0- (x-x_1)/(length)) -1.0/length*(1.0-(x-x_1_prev)/(length_2_prev));		
		}
	}
	
	//in case of linear elements
	else if(p==1){
		//printf("huh");
		//
		//in the same way as the quadratic elements the linear elements are defined. The decreasing basis
		//function with a zero at x2 is defined in h[0] and the increasing basis function with a zero at x1
		//is defined in h[1]
		h[0] = -x/length + x1/length + 1.0;
		h_x[0] = -1.0/length;
		
		h[1] = x/length - x1/length;
		h_x[1] = 1.0/length;
	}

	//Note that this function used the fact that the elements are all identical if the order of the basis function
	//is p there are always p+1 basis functions with support in the element. These basis functions are always the
	//same.
	//printf("%f\n", h[0]);
}


//definition of the potential note that this can be a function or tabulated values which can be interpolated
double phi(double x, double *phi_arr, double *x_nodes, int j){
	
	double length;
	double h[p+1], h_x[p+1];
	//printf("x = %f\n", x);
	if(j < 0){
	if(g_flag == 0){
		j = (int) (x/fabs(x)*(pow(sqrt(fabs(x)) + sqrt(kappa), 2.0) - kappa)/delta);
	}
	else{
		j = (int) (x/dg);
	}
	if(j<0){
		j = 0;
		//length = x_nodes[j+1] - x_nodes[j];
	}
	if(j>=n-1){
		j = n-1;
		//length = x_nodes[j] - x_nodes[j-1];
	}
	}

	//printf("element = %d\n", j);
	linear_element(x_nodes, j, x, h, h_x, length, 0);
	double sum = 0;
	for(int l=0; l<p+1; l++){
		//printf("j = %d, l = %d\n", j, l);
		//printf("phi_arr[j+l] = %f\n", phi_arr[j+l]);
		sum += h[l]*phi_arr[j+l];
	}
	return sum;//3.0/(pow(10.0,5.0))*pow(x-10.0, 5.0);
}

double evaluate_density(double x, double *n_arr, double *x_nodes, int j){
	//int j;
	double length; 
	double h[p+1], h_x[p+1];
	if(j < 0){
	if(g_flag == 0){
		j = (int) (x/fabs(x)*(pow(sqrt(fabs(x)) + sqrt(kappa), 2.0) - kappa)/delta);
	}
	else{
		j = (int) (x/dg);
	}
	//j = (int) j - floor(j/(n-1));
	if(j<0) {
		j = 0;
	}
	if(j>=n-1){
		j = n-1;
	}
	}
	linear_element(x_nodes, j, x, h, h_x, length, 0);
	double sum=0;
	for(int l=0; l<p+1; l++){
		sum += h[l]*n_arr[j+l];
	}
	return sum; 
}

//definition of the gradient of the potential given the potential above
double phi_prime(double x, double *phi_arr, double *x_nodes, int squared_flag, int j){
	double length;
	double h[p+1], h_x[p+1];
	if(j < 0){
	if(g_flag == 0){
		j = (int) (x/fabs(x)*(pow(sqrt(fabs(x)) + sqrt(kappa), 2.0)-kappa)/delta); 
	}
	else{
		if(squared_flag == 1){
			j = (int) (sqrt(x)/dg);
		}
		else{
			j = (int) (x/dg);
		}
	}
	//j = (int) j - floor(j/(n-1));
	//printf("j = %d\n", j);
	//length = x_nodes[j+1] - x_nodes[j];
	//return -phi_arr[j]/length + phi_arr[j+1]/length;
	//printf("x= %f, dx = %f, j = %d\n", x, dx, j);
	
	if(j<0){
		j = 0;
	}
	else if(j>=n-1){
		j = n-1;
	}
	}
	linear_element(x_nodes, j, x, h, h_x, length, squared_flag);
	double sum = 0;
	for(int l=0; l<p+1; l++){
		sum+= h_x[l]*phi_arr[j+l];
	}
	return sum;

	/*		
	for(int j=0; j<n+p-2; j++){
		//printf("index = %d, n_elements = %d, x=%f\n", j, n-1, x);
		if((x>=x_nodes[j]) && (x<=x_nodes[j+1])){
			//printf("index = %d, n_elements = %d, x=%f, x_left = %f, x_right = %f\n", j, n-1, x, x_nodes[j], x_nodes[j+1]);
			//printf("phi_prime = %f\n", -phi_arr[j]/dx + phi_arr[j+1]/dx);
			return -phi_arr[j]/dx + phi_arr[j+1]/dx;
		}
		else if(x<0){
			//printf("%f\n", -phi_arr[0]/dx + phi_arr[1]/dx);
			return -phi_arr[0]/dx + phi_arr[1]/dx;
			//printf("%f\n", -phi_arr[0]/dx + phi_arr[1]/dx;
		}
		else if(x>x_nodes[n-1]){
			//printf("x = %f, phi_prime = %f\n", x, -phi_arr[n+p-3]/dx + phi_arr[n+p-2]/dx);
			return -phi_arr[n+p-3]/dx + phi_arr[n+p-2]/dx; 
			
		}
		//printf("yes2\n");
	}
	*/
	//printf("yes, x= %f\n", x);
	//return 0.0;
	//return 5.0*3.0/(pow(10.0, 5.0))*pow(x-10.0, 4.0);
}

//distribution function at the sheath entrance
double f_inf(double v, double lambda){
	double v_par = v*cos(lambda);
	if (v_par < 0.0){
		return 0.0;
	}
	else{
		return 1.0/(sqrt(2.0)*pow(M_PI, 1.5))*pow(v_par,2.0)*exp(-1.0/2.0*(v*v));
	}
}


double x_traj(double t, double x1, double vx_prime1, double vy1, double v_par1, double slope){
	return -1.0/(2.0*B*Omega)*(cos(alpha)*(-2.0*vx_prime1*B*sin(Omega*t) + 2.0*B*(vy1 - slope/B*cos(alpha))*cos(Omega*t) - 2.0*(vy1 - slope/B*cos(alpha))*B)) - sin(alpha)*(Omega/(2.0*B)*sin(alpha)*slope*t*t + v_par1*t) + x1;
}

double vx_prime_traj(double t, double x1, double vx_prime1, double vy1, double v_par1, double slope){
	return sin(Omega*t)*(slope/B*cos(alpha)*(cos(Omega*t) - 1.0) + vy1) - cos(Omega*t)*(slope/B*cos(alpha)*sin(Omega*t) - vx_prime1);
}

double vy_traj(double t, double x1, double vx_prime1, double vy1, double v_par1, double slope){
	return cos(Omega*t)*(slope/B*cos(alpha)*(cos(Omega*t) - 1.0) + vy1) + sin(Omega*t)*(slope/B*cos(alpha)*sin(Omega*t) - vx_prime1);
}

double v_par_traj(double t, double x1, double vx_prime1, double vy1, double v_par1, double slope){
	return Omega/B*sin(alpha)*slope*t + v_par1;
}

double vx_traj(double t, double x1, double vx_prime1, double vy1, double v_par1, double slope){
	double A_1, A_2;
	A_1 = vy1 - slope/B*cos(alpha);
	A_2 = -vx_prime1;
	return -cos(alpha)*(A_2*cos(Omega*t) - A_1*sin(Omega*t)) - sin(alpha)*(Omega/B*sin(alpha)*slope*t + v_par1);
}

double vx_dot(double t, double x1, double vx_prime1, double vy1, double v_par1, double slope){
	double A_1, A_2;
	A_1 = vy1 - slope/B*cos(alpha);
	A_2 = -vx_prime1;
	return Omega*cos(alpha)*(A_2*sin(Omega*t) + A_1*cos(Omega*t)) - sin(alpha)*sin(alpha)*Omega/B*slope;
}

void root_finder(double *theta, double lambda, double theta_min, double theta_max){
	if(lambda < alpha){
		*theta = 2.0*M_PI;
	}
	else{
		while (fabs(tan(alpha)/tan(lambda)*(*theta-2.0*M_PI-theta_min) + sin(theta_min) - sin(*theta))>1e-10){
			*theta = *theta - (tan(alpha)/tan(lambda)*(*theta-2.0*M_PI-theta_min) + sin(theta_min) - sin(*theta))/(tan(alpha)/tan(lambda)-cos(*theta));
			//printf("theta = %f, f(theta) = %f\n", *theta, tan(alpha)/tan(lambda)*(*theta - 2*M_PI) - sin(*theta));
		}
	}
}
void linspace(double a, double b, int n, double *u){
	double c;
	int i;
		    
	/* step size */
	c = (b - a)/(n - 1);
			    
	/* fill vector */
	for(i = 0; i < n - 1; ++i){
		u[i] = a + i*c;
	}		        
	/* fix last entry to b */
	u[n - 1] = b;
				    
}
				
void construct_matrix(int n_elements, double *x_nodes, gsl_matrix * A, int multiplication_flag){
	double x_1, x_2, x_3, x_4, x_5, mult_1, mult_2, mult_3, mult_4, mult_5;
	double h_1[p+1], h_1x[p+1], h_2[p+1], h_2x[p+1], h_3[p+1], h_3x[p+1], h_4[p+1], h_4x[p+1], h_5[p+1], h_5x[p+1]; 
	int i_elm;
	double A_value;
	double length;
	
	for (i_elm=0; i_elm<n_elements; i_elm++){
		length = x_nodes[i_elm+1] - x_nodes[i_elm]; 
		//x_mid_1 = (length)*(1.0/2.0*1.0/sqrt(3.0)) + (1.0/2.0)*(x_nodes[i_elm+1] + x_nodes[i_elm]);
		//x_mid_2 = (1.0/2.0*(-1.0/sqrt(3.0)))*(length) + (1.0/2.0)*(x_nodes[i_elm+1] + x_nodes[i_elm]);
		//

		x_1 = 1.0/2.0*(x_nodes[i_elm+1]+x_nodes[i_elm]);
		x_2 = length*1.0/2.0*(1.0/3.0*sqrt(5.0-2.0*sqrt(10.0/7.0))) + 1.0/2.0*(x_nodes[i_elm+1] + x_nodes[i_elm]);
		x_3 = length*1.0/2.0*(-1.0/3.0*sqrt(5.0-2.0*sqrt(10.0/7.0))) + 1.0/2.0*(x_nodes[i_elm+1] + x_nodes[i_elm]);
		x_4 = length*1.0/2.0*(1.0/3.0*sqrt(5.0+2.0*sqrt(10.0/7.0))) + 1.0/2.0*(x_nodes[i_elm+1] + x_nodes[i_elm]);
		x_5 = length*1.0/2.0*(-1.0/3.0*sqrt(5.0+2.0*sqrt(10.0/7.0))) + 1.0/2.0*(x_nodes[i_elm+1] + x_nodes[i_elm]);
		//printf("x_1 = %f, x_2 = %f, x_3 = %f, x_4 = %f, x_5 = %f\n", x_1, x_2, x_3, x_4, x_5);
		if(multiplication_flag == 1){
			mult_1 = 2.0*x_1;
			mult_2 = 2.0*x_2;
		       	mult_3 = 2.0*x_3;
			mult_4 = 2.0*x_4;
			mult_5 = 2.0*x_5;	
		}
		else{
			mult_1 = 1.0;
			mult_2 = 1.0;
			mult_3 = 1.0;
			mult_4 = 1.0;
			mult_5 = 1.0;
		}

		//printf("%f, %f\n", x_mid_1, x_mid_2);
		linear_element(x_nodes, i_elm, x_1, h_1, h_1x, length,0);
		linear_element(x_nodes, i_elm, x_2, h_2, h_2x, length,0);
		linear_element(x_nodes, i_elm, x_3, h_3, h_3x, length,0);
		linear_element(x_nodes, i_elm, x_4, h_4, h_4x, length,0);
		linear_element(x_nodes, i_elm, x_5, h_5, h_5x, length,0);
		//printf("%f, %f, %f, %f, %f\n", h_1[0], h_2[0], h_3[0], h_4[0], h_5[0]);		
		//printf("%f, %f\n", h[0], l[0]);
		//write this into matrix definitions
		
		for(int l = 0; l<p+1; l++){
			for(int k = 0; k<p+1; k++){
				A_value = gsl_matrix_get(A, i_elm+l, i_elm+k);
				A_value += length/2.0*(128.0/225.0*h_1[l]*h_1[k]*mult_1 + (322.0+13.0*sqrt(70.0))/900.0*h_2[l]*h_2[k]*mult_2 + (322.0+13.0*sqrt(70.0))/900.0*h_3[l]*h_3[k]*mult_3 + (322.0-13.0*sqrt(70.0))/900.0*h_4[l]*h_4[k]*mult_4 + (322.0-13.0*sqrt(70.0))/900.0*h_5[l]*h_5[k]*mult_5); 
				gsl_matrix_set(A, i_elm+l, i_elm+k, A_value);
			}
		}

	}
	//PRINTING MATRIX FOR CHECKING
	/*	
	for(int l = 0; l<n+p-1; l++){
		for(int k = 0; k<n+p-1; k++){
			if(k != n+p-2){
				printf("%.4f ", gsl_matrix_get(A, l, k));
			}
			else{
				printf("%.4f\n", gsl_matrix_get(A, l, k));
			}
		}
	}
	*/	
	//gsl_matrix_set(A, n_elements, n_elements, 1.0);
	
	//printf("A_00 = %f, A_01 = %f, A_02 = %f, A_10 = %f, A_11 = %f, A_12 = %f, A20 = %f, A_21 = %f, A22 = %f\n", gsl_matrix_get(A, 0, 0), gsl_matrix_get(A, 0, 1), gsl_matrix_get(A, 0, 2), gsl_matrix_get(A, 1, 0), gsl_matrix_get(A, 1, 1), gsl_matrix_get(A, 1, 2), gsl_matrix_get(A, 2, 0), gsl_matrix_get(A, 2, 1), gsl_matrix_get(A, 2, 2));
}

void construct_matrix_D(int n_elements, double *x_nodes, gsl_matrix * A, double *phi_arr, double gamma){
	double x_1, x_2, x_3, x_4, x_5, mult_1, mult_2, mult_3, mult_4, mult_5;
	double h_1[p+1], h_1x[p+1], h_2[p+1], h_2x[p+1], h_3[p+1], h_3x[p+1], h_4[p+1], h_4x[p+1], h_5[p+1], h_5x[p+1]; 
	int i_elm;
	double A_value;
	double length;
	
	for (i_elm=0; i_elm<n_elements; i_elm++){
		length = x_nodes[i_elm+1] - x_nodes[i_elm]; 
		//x_mid_1 = (length)*(1.0/2.0*1.0/sqrt(3.0)) + (1.0/2.0)*(x_nodes[i_elm+1] + x_nodes[i_elm]);
		//x_mid_2 = (1.0/2.0*(-1.0/sqrt(3.0)))*(length) + (1.0/2.0)*(x_nodes[i_elm+1] + x_nodes[i_elm]);
		//

		x_1 = 1.0/2.0*(x_nodes[i_elm+1]+x_nodes[i_elm]);
		x_2 = length*1.0/2.0*(1.0/3.0*sqrt(5.0-2.0*sqrt(10.0/7.0))) + 1.0/2.0*(x_nodes[i_elm+1] + x_nodes[i_elm]);
		x_3 = length*1.0/2.0*(-1.0/3.0*sqrt(5.0-2.0*sqrt(10.0/7.0))) + 1.0/2.0*(x_nodes[i_elm+1] + x_nodes[i_elm]);
		x_4 = length*1.0/2.0*(1.0/3.0*sqrt(5.0+2.0*sqrt(10.0/7.0))) + 1.0/2.0*(x_nodes[i_elm+1] + x_nodes[i_elm]);
		x_5 = length*1.0/2.0*(-1.0/3.0*sqrt(5.0+2.0*sqrt(10.0/7.0))) + 1.0/2.0*(x_nodes[i_elm+1] + x_nodes[i_elm]);
		//printf("x_1 = %f, x_2 = %f, x_3 = %f, x_4 = %f, x_5 = %f\n", x_1, x_2, x_3, x_4, x_5);
		

		//printf("%f, %f\n", x_mid_1, x_mid_2);
		linear_element(x_nodes, i_elm, x_1, h_1, h_1x, length,0);
		linear_element(x_nodes, i_elm, x_2, h_2, h_2x, length,0);
		linear_element(x_nodes, i_elm, x_3, h_3, h_3x, length,0);
		linear_element(x_nodes, i_elm, x_4, h_4, h_4x, length,0);
		linear_element(x_nodes, i_elm, x_5, h_5, h_5x, length,0);
		//printf("%f, %f, %f, %f, %f\n", h_1[0], h_2[0], h_3[0], h_4[0], h_5[0]);		
		//printf("%f, %f\n", h[0], l[0]);
		//write this into matrix definitions
		
		for(int l = 0; l<p+1; l++){
			for(int k = 0; k<p+1; k++){
				A_value = gsl_matrix_get(A, i_elm+l, i_elm+k);
				printf("set_fault_D\n");
				A_value += gamma*gamma*length/2.0*(128.0/225.0*h_1x[l]*h_1x[k] + (322.0+13.0*sqrt(70.0))/900.0*h_2x[l]*h_2x[k] + (322.0+13.0*sqrt(70.0))/900.0*h_3x[l]*h_3x[k] + (322.0-13.0*sqrt(70.0))/900.0*h_4x[l]*h_4x[k] + (322.0-13.0*sqrt(70.0))/900.0*h_5x[l]*h_5x[k]) + length/2.0*(128.0/225.0*h_1[l]*h_1[k]*exp(phi(x_1, phi_arr, x_nodes, i_elm)) + (322.0+13.0*sqrt(70.0))/900.0*h_2[l]*h_2[k]*exp(phi(x_2, phi_arr, x_nodes, i_elm)) + (322.0+13.0*sqrt(70.0))/900.0*h_3[l]*h_3[k]*exp(phi(x_3, phi_arr, x_nodes, i_elm)) + (322.0-13.0*sqrt(70.0))/900.0*h_4[l]*h_4[k]*exp(phi(x_4, phi_arr, x_nodes, i_elm)) + (322.0-13.0*sqrt(70.0))/900.0*h_5[l]*h_5[k]*exp(phi(x_5, phi_arr, x_nodes, i_elm)));
				//A_value += length/2.0*(128.0/225.0*h_1x[l]*h_1x[k]*exp(-phi(x_1, phi_arr, x_nodes)) + (322.0+13.0*sqrt(70.0))/900.0*h_2x[l]*h_2x[k]*exp(-phi(x_2, phi_arr, x_nodes)) + (322.0+13.0*sqrt(70.0))/900.0*h_3x[l]*h_3x[k]*exp(-phi(x_3, phi_arr, x_nodes)) + (322.0-13.0*sqrt(70.0))/900.0*h_4x[l]*h_4x[k]*exp(-phi(x_4, phi_arr, x_nodes)) + (322.0-13.0*sqrt(70.0))/900.0*h_5x[l]*h_5x[k]*exp(-phi(x_5, phi_arr, x_nodes))); 
				gsl_matrix_set(A, i_elm+l, i_elm+k, A_value);
			}
		}

	}
	//PRINTING MATRIX FOR CHECKING
/*	
	for(int l = 0; l<n+p-1; l++){
		for(int k = 0; k<n+p-1; k++){
			if(k != n+p-2){
				printf("%.4f ", gsl_matrix_get(A, l, k));
			}
			else{
				printf("%.4f\n", gsl_matrix_get(A, l, k));
			}
		}
	}
*/		
	//gsl_matrix_set(A, n_elements, n_elements, 1.0);
	
	//printf("A_00 = %f, A_01 = %f, A_02 = %f, A_10 = %f, A_11 = %f, A_12 = %f, A20 = %f, A_21 = %f, A22 = %f\n", gsl_matrix_get(A, 0, 0), gsl_matrix_get(A, 0, 1), gsl_matrix_get(A, 0, 2), gsl_matrix_get(A, 1, 0), gsl_matrix_get(A, 1, 1), gsl_matrix_get(A, 1, 2), gsl_matrix_get(A, 2, 0), gsl_matrix_get(A, 2, 1), gsl_matrix_get(A, 2, 2));
}


void make_vector_2(int n_elements, gsl_vector* rhs_2, double *phi_arr, double *n_arr, double *x_nodes){
	double x_1, x_2, x_3, x_4, x_5;
	double ev_1, ev_2, ev_3, ev_4, ev_5;
	double h_1[p+1], hx_1[p+1], h_2[p+1], hx_2[p+1], h_3[p+1], hx_3[p+1], h_4[p+1], hx_4[p+1], h_5[p+1], hx_5[p+1];
	int i_elm;
	double length;
	double rhs_value;
	for(i_elm=0; i_elm < n_elements; i_elm++){
		length = x_nodes[i_elm+1] - x_nodes[i_elm]; 
		//x_mid_1 = (length)*(1.0/2.0*1.0/sqrt(3.0)) + (1.0/2.0)*(x_nodes[i_elm+1] + x_nodes[i_elm]);
		//x_mid_2 = (1.0/2.0*(-1.0/sqrt(3.0)))*(length) + (1.0/2.0)*(x_nodes[i_elm+1] + x_nodes[i_elm]);
		//

		x_1 = 1.0/2.0*(x_nodes[i_elm+1]+x_nodes[i_elm]);
		x_2 = length*1.0/2.0*(1.0/3.0*sqrt(5.0-2.0*sqrt(10.0/7.0))) + 1.0/2.0*(x_nodes[i_elm+1] + x_nodes[i_elm]);
		x_3 = length*1.0/2.0*(-1.0/3.0*sqrt(5.0-2.0*sqrt(10.0/7.0))) + 1.0/2.0*(x_nodes[i_elm+1] + x_nodes[i_elm]);
		x_4 = length*1.0/2.0*(1.0/3.0*sqrt(5.0+2.0*sqrt(10.0/7.0))) + 1.0/2.0*(x_nodes[i_elm+1] + x_nodes[i_elm]);
		x_5 = length*1.0/2.0*(-1.0/3.0*sqrt(5.0+2.0*sqrt(10.0/7.0))) + 1.0/2.0*(x_nodes[i_elm+1] + x_nodes[i_elm]);
		linear_element(x_nodes, i_elm, x_1, h_1, hx_1, length,0);
		linear_element(x_nodes, i_elm, x_2, h_2, hx_2, length,0);
		linear_element(x_nodes, i_elm, x_3, h_3, hx_3, length,0);
		linear_element(x_nodes, i_elm, x_4, h_4, hx_4, length,0);
		linear_element(x_nodes, i_elm, x_5, h_5, hx_5, length,0);
		
		//printf("dens = %f, phi = %f, x1 = %f\n", evaluate_density(x_1, n_arr, x_nodes, i_elm), phi(x_1, phi_arr, x_nodes, i_elm), x_1);
		//printf("dens = %f, phi = %f, x2 = %f\n", evaluate_density(x_2, n_arr, x_nodes, i_elm), phi(x_2, phi_arr, x_nodes, i_elm), x_2);
		//printf("dens = %f, phi = %f, x3 = %f\n", evaluate_density(x_3, n_arr, x_nodes, i_elm), phi(x_3, phi_arr, x_nodes, i_elm), x_3);
		//printf("dens = %f, phi = %f, x4 = %f\n", evaluate_density(x_4, n_arr, x_nodes, i_elm), phi(x_4, phi_arr, x_nodes, i_elm), x_4);
		//printf("dens = %f, phi = %f, x5 = %f\n", evaluate_density(x_5, n_arr, x_nodes, i_elm), phi(x_5, phi_arr, x_nodes, i_elm), x_5);
		//printf("seg_fault_rhs_2\n");
		
		ev_1 = exp(phi(x_1, phi_arr, x_nodes, i_elm))*(phi(x_1, phi_arr, x_nodes, i_elm) + exp(-phi(x_1, phi_arr, x_nodes, i_elm))*(evaluate_density(x_1, n_arr, x_nodes, i_elm) - exp(phi(x_1, phi_arr, x_nodes, i_elm))));
		ev_2 = exp(phi(x_2, phi_arr, x_nodes, i_elm))*(phi(x_2, phi_arr, x_nodes, i_elm) + exp(-phi(x_2, phi_arr, x_nodes, i_elm))*(evaluate_density(x_2, n_arr, x_nodes, i_elm) - exp(phi(x_2, phi_arr, x_nodes, i_elm))));
		ev_3 = exp(phi(x_3, phi_arr, x_nodes, i_elm))*(phi(x_3, phi_arr, x_nodes, i_elm) + exp(-phi(x_3, phi_arr, x_nodes, i_elm))*(evaluate_density(x_3, n_arr, x_nodes, i_elm) - exp(phi(x_3, phi_arr, x_nodes, i_elm))));
		ev_4 = exp(phi(x_4, phi_arr, x_nodes, i_elm))*(phi(x_4, phi_arr, x_nodes, i_elm) + exp(-phi(x_4, phi_arr, x_nodes, i_elm))*(evaluate_density(x_4, n_arr, x_nodes, i_elm) - exp(phi(x_4, phi_arr, x_nodes, i_elm))));
		ev_5 = exp(phi(x_5, phi_arr, x_nodes, i_elm))*(phi(x_5, phi_arr, x_nodes, i_elm) + exp(-phi(x_5, phi_arr, x_nodes, i_elm))*(evaluate_density(x_5, n_arr, x_nodes, i_elm) - exp(phi(x_5, phi_arr, x_nodes, i_elm))));

		//printf("1 = %f, 2=%f, 3=%f, 4=%f, 5=%f\n", ev_1, ev_2, ev_3, ev_4, ev_5);
		//sleep(3);	
		for(int k=0; k<p+1; k++){
			rhs_value = gsl_vector_get(rhs_2, i_elm+k);
			rhs_value += length/2.0*(128.0/225.0*ev_1*h_1[k] + (322.0 + 13.0*sqrt(70.0))/900.0*ev_2*h_2[k] + (322.0+13.0*sqrt(70.0))/900.0*ev_3*h_3[k] + (322.0-13.0*sqrt(70.0))/900.0*ev_4*h_4[k] + (322.0-13.0*sqrt(70.0))/900.0*ev_5*h_5[k]);
			//rhs_value += length/2.0*(128.0/225.0*ev_1*h_1[k]*2.0*x_1 + (322.0 + 13.0*sqrt(70.0))/900.0*ev_2*h_2[k]*2.0*x_2 + (322.0+13.0*sqrt(70.0))/900.0*ev_3*h_3[k]*2.0*x_3 + (322.0-13.0*sqrt(70.0))/900.0*ev_4*h_4[k]*2.0*x_4 + (322.0-13.0*sqrt(70.0))/900.0*ev_5*h_5[k]*2.0*x_5);
			gsl_vector_set(rhs_2, i_elm+k, rhs_value);
		}

	}
}

void set_boundary_conditions(gsl_vector *rhs_2, gsl_matrix *A_2, double phi_0, double phi_inf){
	gsl_vector_set(rhs_2, 0, phi_0);
	gsl_vector_set(rhs_2, n+p-2, phi_inf);
	gsl_matrix_set(A_2, 0, 0, 1.0);
	gsl_matrix_set(A_2, n+p-2, n+p-2, 1.0);

	for(int j = 1; j < p+1; j++){
		gsl_matrix_set(A_2, 0, j, 0.0);
		gsl_matrix_set(A_2, n+p-2, n+p-2-j, 0.0);
	}
	//PRINTING MATRIX FOR CHECKING
	
	for(int l = 0; l<n+p-1; l++){
		for(int k = 0; k<n+p-1; k++){
			if(k != n+p-2){
				printf("%.4f ", gsl_matrix_get(A_2, l, k));
			}
			else{
				printf("%.4f\n", gsl_matrix_get(A_2, l, k));
			}
		}
	}
	
}

double root_finder_vx(double t_left, double t_right, double x1, double vx_prime1, double vy1, double v_par1, double slope){
	int j, j_max;
	double df, delta_x, f, rtn;
	j_max = 100;

	rtn = 0.5*(t_left + t_right);
	//printf("rtn = %f\n", rtn);
	for(j = 1; j<=j_max; j++){
		f = vx_traj(rtn, x1, vx_prime1, vy1, v_par1, slope);
		df = vx_dot(rtn, x1, vx_prime1, vy1, v_par1, slope);
		delta_x = f/df;
		rtn -= delta_x;
		//printf("f = %f, df = %f, delta_x = %f, rtn = %f\n", f, df, delta_x, rtn);
		if((t_left - rtn)*(rtn - t_right) < 0.0){
			//printf("Jumped out of brackets in root_finder_vx\n");
			if(rtn < t_left){
				rtn = (t_left + rtn + delta_x)*0.5;
			}
			else if(rtn > t_right){
				rtn = (t_right + rtn + delta_x)*0.5;
			}
			else{
				printf("something is wrong\n");
			}
			//printf("t_left = %f, t_right = %f, vx(t_left) = %f, vx(t_right) = %f, vx(rtn) = %f, rtn = %f, delta_x = %f, f = %f, df = %f\n", t_left, t_right, vx_traj(t_left, x1, vx_prime1, vy1, v_par1, slope), vx_traj(t_right, x1, vx_prime1, vy1, v_par1, slope), vx_traj(rtn, x1, vx_prime1, vy1, v_par1, slope), rtn, delta_x, f, df);
		}
		if(fabs(f) < 1e-13) return rtn;
	}
	printf("Maximum number of iterations exceeded in root_finder_vx\n");
	return 0.0;
}

double root_finder_x(double t_left, double t_right, double x_out, double x1, double vx_prime1, double vy1, double v_par1, double slope){
	int j, j_max;
	double df, delta_x, f, rtn;

	j_max = 100;
	
	rtn = 0.5*(t_left + t_right);
	for(j = 1; j<=j_max; j++){
		f = x_traj(rtn, x1, vx_prime1, vy1, v_par1, slope) - x_out; 
		df = vx_traj(rtn, x1, vx_prime1, vy1, v_par1, slope);
		delta_x = f/df;
		rtn -= delta_x;
		if((t_left - rtn)*(rtn - t_right) < 0.0){
			//printf("Jumped out of brackets in root_finder_x, with rtn = %f\n", rtn);
			if(rtn < t_left){
				rtn = (t_left + rtn + delta_x)*0.5;
			}
			else if(rtn > t_right){
				rtn = (t_right + rtn + delta_x)*0.5;
			}
			else{
				printf("something is wrong\n");
			}
			//printf("t_left = %f, t_right = %f, x(t_left) = %f, x(t_right) = %f, x(rtn) = %f, rtn = %f, delta_x = %f, f = %f, df = %f\n", t_left, t_right, x_traj(t_left, x1, vx_prime1, vy1, v_par1, slope) - x_out, x_traj(t_right, x1, vx_prime1, vy1, v_par1, slope) - x_out, x_traj(rtn, x1, vx_prime1, vy1, v_par1, slope) - x_out, rtn, delta_x, f, df);
		}
		if(fabs(f) < 1e-13) return rtn;
	}
	printf("Maximum number of iterations exceeded in root_finder_x\n");
	printf("t_left = %f, t_right = %f, x_out = %f, x1 = %f, vx_prime1 = %f, vy1 = %f, v_par1 = %f, slope = %f, rtn = %.15f, f = %.15f df = %.15f, deltax_x = %.15f\n", t_left, t_right, x_out, x1, vx_prime1, vy1, v_par1, slope, rtn, f, df, delta_x);
	return 0.0;
}

double find_turning_point(double t_left, double t_right, double x1, double vx_prime1, double vy1, double v_par1, double slope){
	return root_finder_vx(t_left, t_right, x1, vx_prime1, vy1, v_par1, slope);
}

double find_x_crossing(double t_left, double t_right, double x_out, double x1, double vx_prime1, double vy1, double v_par1, double slope){
	if(t_right == t_left){
		int n = 0;
		double t_right_before = t_right;
		while((x_traj(t_right, x1, vx_prime1, vy1, v_par1, slope) - x_out)/(x_traj(t_left, x1, vx_prime1, vy1, v_par1, slope) - x_out) > 0.0){
			n += 1;
			t_left = t_right;
			t_right += 2.0*M_PI/Omega*0.5;
			//printf("wow, t_right= %f, t_right_before = %f, n = %d\n", t_right, t_right_before, n);
			//sleep(1);
			//printf("x_out = %f, x1 = %f, vx_prime1 = %f, vy1 = %f, v_par1 = %f, slope = %f\n", x_out, x1, vx_prime1, vy1, v_par1, slope);
		}
	}
	return root_finder_x(t_left, t_right, x_out, x1, vx_prime1, vy1, v_par1, slope);
}

void find_vx_dot_zeros(double t1, double x1, double vx_prime1, double vy1, double v_par1, double slope, double *t_left, double *t_right){
	double mult_factor = -1.0;
	double A_1 = vy1 - slope/B*cos(alpha);
	double A_2 = -vx_prime1;
	double t_1 = -1/Omega*(asin(-(sin(alpha)*tan(alpha)/B*slope)/sqrt(A_1*A_1 + A_2*A_2)) + atan(A_1/A_2) - M_PI);
	double t_2 = 1/Omega*(asin(-(sin(alpha)*tan(alpha)/B*slope)/sqrt(A_1*A_1 + A_2*A_2)) - atan(A_1/A_2));
	//printf("A_1 = %f, A_2 = %f\n", A_1, A_2);
	//printf("x1 = %f, vx_prime1 = %f, vy1 = %f, v_par1 = %f, slope = %f\n", x1, vx_prime1, vy1, v_par1, slope);
	//printf("t_1 = %f, t_2 = %f\n", t_1, t_2);
	if((fabs(vx_dot(t_1, x1, vx_prime1, vy1, v_par1, slope)) > 1e-10) || (fabs(vx_dot(t_2, x1, vx_prime1, vy1, v_par1, slope)) > 1e-10)){
		mult_factor = 1.0;
		t_1 = -1/Omega*(asin((sin(alpha)*tan(alpha)/B*slope)/sqrt(A_1*A_1 + A_2*A_2)) + atan(A_1/A_2) - M_PI);
		t_2 = 1/Omega*(asin((sin(alpha)*tan(alpha)/B*slope)/sqrt(A_1*A_1 + A_2*A_2)) - atan(A_1/A_2));

	}
	int n=0;
	while(t_1 - t1 < 0.0){
		n+=1;
		t_1 = -1/Omega*(asin(mult_factor*(sin(alpha)*tan(alpha)/B*slope)/sqrt(A_1*A_1 + A_2*A_2)) + atan(A_1/A_2) - M_PI - ((double) (n))*2.0*M_PI);
		//printf("t_1_check, t_1, t_1 = %f, t1 = %f, %d\n", t_1, t1, n);
		//printf("t1 = %f, x1 = %f, vx_prime1 = %f, vy1 = %f, v_par1 = %f, slope = %f\n", t1, x1, vx_prime1, vy1, v_par1, slope);
	}
	n = 0;
	while(t_2 - t1 < 0.0){
		n+=1;
		t_2 = 1/Omega*(asin(mult_factor*(sin(alpha)*tan(alpha)/B*slope)/sqrt(A_1*A_1 + A_2*A_2)) - atan(A_1/A_2) + ((double) (n))*2.0*M_PI);
		//printf("t_2_check, t_2 = %f, %d\n", t_2, n);
	}
	//printf("t_2 = %f\n", t_2);
	if(t_2 < t_1){
		double t_3 = t_1; 
		t_1 = t_2; 
		t_2 = t_3;
	}
	//printf("t_1 = %f, t_2 = %f\n", t_1, t_2);
	if(fabs(vx_traj(t1, x1, vx_prime1, vy1, v_par1, slope)) < 1e-10){
		if(vx_traj(t_2, x1, vx_prime1, vy1, v_par1, slope)/vx_traj(t_1, x1, vx_prime1, vy1, v_par1, slope) < 0.0){
			*t_left = t_1; 
			*t_right = t_2;
			return;
		}
		else{
			*t_left = t1;
			*t_right = t1;
			return;
		}
	}
	if(vx_dot(t1, x1, vx_prime1, vy1, v_par1, slope)/vx_traj(t1, x1, vx_prime1, vy1, v_par1, slope) < 0.0){
		if(vx_traj(t1, x1, vx_prime1, vy1, v_par1, slope)/vx_traj(t_1, x1, vx_prime1, vy1, v_par1, slope) < 0.0){
			*t_left = t1;
			*t_right = t_1;
			return;
		}	
		else{
			*t_left = t1;
			*t_right = t1;
			return;
		}
	}
	else{
		//printf("vx_dot(t1, x1, vx_prime1, vy1, v_par1, slope) = %f\n", vx_dot(t1, x1, vx_prime1, vy1, v_par1, slope));
		//printf("vx_dot(t_1) = %f, vx_dot(t_2) = %f\n", vx_dot(t_1, x1, vx_prime1, vy1, v_par1, slope), vx_dot(t_2, x1, vx_prime1, vy1, v_par1, slope));
		//printf("hey\n");
		//printf("vx(t_1) = %f, vx(0) = %f\n", vx_traj(t_1, x1, vx_prime1, vy1, v_par1, slope), vx_traj(t1, x1, vx_prime1, vy1, v_par1, slope));
		if(vx_traj(t_2, x1, vx_prime1, vy1, v_par1, slope)/vx_traj(t_1, x1, vx_prime1, vy1, v_par1, slope) < 0.0){
			//printf("vx(t_2) = %f, vx(t_1) = %f\n", vx_traj(t_2, x1, vx_prime1, vy1, v_par1, slope), vx_traj(t_1, x1, vx_prime1, vy1, v_par1, slope));
			*t_left = t_1;
			*t_right = t_2;
			return;
		}
		else{
			*t_left = t1;
			*t_right = t1;
			return;
		}
	}
}

void find_turning_repeatedly(double t1, double x1, double vx_prime1, double vy1, double v_par1, double slope, double *t_crossing, int *next_elm, double *x_crossing, int i_element, double *x_nodes, int *iterations){
	double t_left, t_right;
	double x_out;
	double t_turning;
	find_vx_dot_zeros(t1, x1, vx_prime1, vy1, v_par1, slope, &t_left, &t_right);
	//printf("t_left = %f, t_right=%f\n", t_left, t_right);
	if(t_left == t_right){
		if(vx_traj(t1, x1, vx_prime1, vy1, v_par1, slope) > 0.001){
			//printf("this is weird, vx0 = %f, t0 = %f\n", vx_traj(t1, x1, vx_prime1, vy1, v_par1, slope), t1);
			x_out = x_nodes[i_element+1];
			*next_elm = i_element+1;
		}
		else{
			x_out = x_nodes[i_element];
			*next_elm = i_element-1;
		}
		//printf("t_left == t_right\n");
		*t_crossing = find_x_crossing(t_left, t_right, x_out, x1, vx_prime1, vy1, v_par1, slope);
		//printf("passed\n");
		*x_crossing = x_out;
		return;
	}
	else{
		t_turning = find_turning_point(t_left, t_right, x1, vx_prime1, vy1, v_par1, slope);
		if(x_traj(t_turning, x1, vx_prime1, vy1, v_par1, slope) > x_nodes[i_element+1]){
			x_out = x_nodes[i_element+1];
			*t_crossing = find_x_crossing(t1, t_turning, x_out, x1, vx_prime1, vy1, v_par1, slope);
			*next_elm = i_element + 1;
			*x_crossing = x_out;
			return;
		}
		else if(x_traj(t_turning, x1, vx_prime1, vy1, v_par1, slope) < x_nodes[i_element]){
			x_out = x_nodes[i_element];
			*t_crossing = find_x_crossing(t1, t_turning, x_out, x1, vx_prime1, vy1, v_par1, slope);
			*next_elm = i_element-1;
			*x_crossing = x_out;
			return;
		}
		else{
			*iterations += 1;
			if(*iterations > 100){
				printf("balbal");
			}
			find_turning_repeatedly(t_turning, x1, vx_prime1, vy1, v_par1, slope, t_crossing, next_elm, x_crossing, i_element, x_nodes, iterations);
			return;
		}
	}
}

void find_t_next(double t1, double *x1, double vx_prime1, double vy1, double v_par1, double slope, double *time_step, int *i_element, double *x_nodes){
	int next_elm;
	double vx;
	int iterations = 0;
	double t_turning;
	double t_crossing, x_crossing;
	double A_1, A_2, E_term, B_term;
	A_1 = vy1 - slope/B*cos(alpha);
	A_2 = -vx_prime1;
	E_term = fabs(sin(alpha)*tan(alpha)*Omega/B*slope);
	B_term = fabs(sqrt(A_2*A_2 + A_1*A_1));

	if(E_term > B_term){
		//printf("E_term > B_term");
		vx = vx_traj(t1, *x1, vx_prime1, vy1, v_par1, slope);
		if(vx > 0){
			t_turning = find_turning_point(t1, t1+2.0*M_PI/Omega*0.5, *x1, vx_prime1, vy1, v_par1, slope);
			if(x_traj(t_turning, *x1, vx_prime1, vy1, v_par1, slope) > x_nodes[*i_element+1]){
				x_crossing = x_nodes[*i_element+1];
				t_crossing = find_x_crossing(t1, t_turning, x_crossing, *x1, vx_prime1, vy1, v_par1, slope);
				next_elm = *i_element+1;
			}
			else{
				x_crossing = x_nodes[*i_element];
				t_crossing = find_x_crossing(t1, t_turning, x_crossing, *x1, vx_prime1, vy1, v_par1, slope);
				next_elm = *i_element-1;
			}
		}
		else{
			x_crossing = x_nodes[*i_element];
			t_crossing = find_x_crossing(t1, t1, x_crossing, *x1, vx_prime1, vy1, v_par1, slope);
			next_elm = *i_element-1;
		}
	}
	else{
		find_turning_repeatedly(t1, *x1, vx_prime1, vy1, v_par1, slope, &t_crossing, &next_elm, &x_crossing, *i_element, x_nodes, &iterations);	
	}
	*i_element = next_elm;
	*x1 = x_crossing;
	*time_step = t_crossing;
} 


void calculate_contribution(double tau, double x0, double x1, double x2, double vx_prime1, double vy1, double v_par1, double slope, gsl_vector *rhs, double a, int i_elm){
	double added_value_to_integral_0, added_value_to_integral_1;
	double integral_value_0, integral_value_1;	
	double length;
	//printf("%d, %d\n", i_elm, i_elm_prev);
	double A_1 = vy1 - slope/B*cos(alpha);
	double A_2 = -vx_prime1;
	double h = fabs(x2 - x1);
	//printf("x1 = %f, x0 = %f, h = %f\n", x1, x0, h);
	//sleep(1);

	added_value_to_integral_0 = tau/h*(x1-x0) + sin(alpha)*tau*tau*v_par1/(2.0*h) + 1.0/(h*Omega*Omega)*(A_1*cos(alpha)*sin(Omega*tau) - A_2*cos(alpha)*cos(Omega*tau) + A_2*cos(alpha)) + sin(alpha)*sin(alpha)*slope*Omega*tau*tau*tau/(6.0*B*h) + (h*Omega - A_1*cos(alpha))*tau/(h*Omega);
	added_value_to_integral_1 = -added_value_to_integral_0 + tau;
	
	integral_value_0 = gsl_vector_get(rhs, i_elm);
	integral_value_1 = gsl_vector_get(rhs, i_elm+1);
	//if(i_elm + 1 == 105 || i_elm == 105){
	//	printf("int_value_0_before = %.15f, int_value_1_before = %.15f, i_elm = %d\n", integral_value_0, integral_value_1, i_elm);
	//}
	//if(i_elm + 1 == 104 || i_elm == 104){
	//	printf("int_value_0_before = %.15f, int_value_1_before = %.15f, i_elm = %d\n", integral_value_0, integral_value_1, i_elm);
	//}
	integral_value_0 += a*added_value_to_integral_0;
	integral_value_1 += a*added_value_to_integral_1;
	gsl_vector_set(rhs, i_elm, integral_value_0);
	gsl_vector_set(rhs, i_elm+1, integral_value_1);
	//if(i_elm+1 == 105 || i_elm == 105){
	//	printf("int_value_0_after = %.15f, int_value_1_after = %.15f\n, added_value_to_integral_0 = %f, added_value_to_integral_1 = %f\n", integral_value_0, integral_value_1, added_value_to_integral_0, added_value_to_integral_1);
	//}
	//if(i_elm + 1 == 104 || i_elm == 104){
	//	printf("int_value_0_after = %.15f, int_value_1_after = %.15f\n, added_value_to_integral_0 = %f, added_value_to_integral_1 = %f\n", integral_value_0, integral_value_1, added_value_to_integral_0, added_value_to_integral_1);
	//}
}

double calculate_f(double x0, double v0, double phi0, double theta0, double t0, double dt, double *x_arr, double *phi_arr, double L){
	double x=x0;
//	double Kx0 = phi(0.0, phi_arr, x_arr) - phi(sqrt(0.01*vx0*dt), phi_arr, x_arr) + 1.0/2.0*vx0*vx0 + vy0*Omega*vx0*0.01*dt*cos(alpha)*B;
//	if(Kx0 < 0){
//		return 0.0;
//	}
//	x = x0 + vx0*0.01*dt;
//	vx0 = -sqrt(2.0*Kx0);
	//x = x0 + vx0*dt;
	//
	if(v0 == 0.0){
		return 0.0;
	}
	double vx0 = -v0*sin(theta0);
	double vy0 = v0*cos(theta0)*sin(phi0);
	double vz0 = v0*cos(theta0)*cos(phi0);
	double vx_prime = vx0*cos(alpha) + vz0*sin(alpha);
	double v_par = -vx0*sin(alpha) + vz0*cos(alpha);
	double vy = vy0;
	double t=t0;
	double vx_at_entrance;
	double dt_for_last_step;
	int i_element_previous;
	double x_previous;
	double vy_previous;
	double v_par_previous;
	double vx_prime_previous;
	double time_step;
	double slope;
	//if(fabs(vx0) < 1){
	//	printf("x = %f, v_par = %f, vx_prime = %f, vy = %f, vx = %f\n", x, v_par, vx_prime, vy, vx_prime*cos(alpha) - v_par*sin(alpha));
	//	sleep(2);
	//}
	int i_element = 0; // this is not general

	while (x<L){
		//runge_kutta(&x,&v_par,&vx_prime,&vy,&t, dt, phi_arr, x_arr,1);
		double dx = x_arr[i_element + 1] - x_arr[i_element];
		//printf("x_arr[i_element + 1] = %f, x_arr[i_element] = %f, i_element = %d, i_element+1=%d\n", x_arr[i_element+1], x_arr[i_element], i_element, i_element+1);
		double v_max = sqrt(vx_prime*vx_prime + v_par*v_par + vy*vy + 2.0*phi_arr[i_element+1] - 2.0*phi_arr[i_element]);
		if(x!=x){
			printf("time_step = %.10f, v_max = %.10f, dx = %.10f\n", time_step, v_max, dx);
			sleep(10);
		}
		time_step = -(dx/v_max)/5.0;
		//printf("time_step = %.10f, v_max = %.10f, dx = %.10f\n", time_step, v_max, dx);
	
		i_element_previous = i_element;
		x_previous = x;
		vx_prime_previous = vx_prime; 
		vy_previous = vy;
		v_par_previous = v_par;
		//x_right = position_arr[i_element+1];
		//x_left = position_arr[i_element];
		slope = (phi_arr[i_element+1] - phi_arr[i_element])/(x_arr[i_element+1] - x_arr[i_element]);
		//printf("v0 = %f, theta0 = %f, lambda0 = %f\n", v0, theta0, lambda0);
		//find_t_next(0.0, &x, vx_prime, vy, v_par, slope, &time_step, &i_element, position_arr);
		//if((i_element == number_of_elements) || (i_element == -1)){
		//	break;
		//}
		//double x_traj(double t, double x1, double vx_prime1, double vy1, double v_par1, double slope)
		t += time_step;
		x = x_traj(time_step, x_previous, vx_prime_previous, vy_previous, v_par_previous, slope);
		vx_prime = vx_prime_traj(time_step, x_previous, vx_prime_previous, vy_previous, v_par_previous, slope);
	       	vy = vy_traj(time_step, x_previous, vx_prime_previous, vy_previous, v_par_previous, slope);
		v_par = v_par_traj(time_step, x_previous, vx_prime_previous, vy_previous, v_par_previous, slope);
		if(x > x_arr[i_element+1]){
			i_element = i_element + 1;
		}
		else if(x < x_arr[i_element]){
			i_element = i_element -1;
		}
	//	if(fabs(vx0) < 1){
	//		printf("x = %f, v_par = %f, vx_prime = %f, vy = %f, vx = %f\n", x, v_par, vx_prime, vy, vx_prime*cos(alpha) - v_par*sin(alpha));
		
	//		if(x < 0.1){
	//			sleep(2);
	//		}
	//	}
		if(x<0){
			return 0;
		}
	}
	//printf("x = %.10f, v_par=%.10f, vx_prime = %.10f, vy = %.10f, L = %f\n", x, v_par, vx_prime, vy, L);
	//printf("x = %.10f, v_par=%.10f, vx_prime = %.10f, vy = %.10f, L = %f\n", x, v_par, vx_prime, vy, L);
	x = x_previous;
	vx_prime = vx_prime_previous;
	vy = vy_previous;
	v_par = v_par_previous;
	
	//printf("x = %.10f, v_par=%.10f, vx_prime = %.10f, vy = %.10f, L = %f\n", x, v_par, vx_prime, vy, L);
	//sleep(2);
	//runge_kutta(&x, &v_par, &vx_prime, &vy, &t, -dt, phi_arr, x_arr, 1);
	while(x < L-1e-10){
		x_previous = x;
		vx_prime_previous = vx_prime; 
		vy_previous = vy;
		v_par_previous = v_par;
		if(g_flag == 0){
			vx_at_entrance = sqrt((pow(vx_prime,2.0)+pow(v_par,2.0) + pow(vy,2.0)) - 2.0*slope*(L-x));
		}
		//else{
		//	vx_at_entrance = 2.0*sqrt(1.0/2.0*(pow(vx_prime,2.0)+pow(v_par,2.0) + pow(vy,2.0)) - Z*e*phi(sqrt(x), phi_arr, x_arr)/m_i);
		//}
		dt_for_last_step = -fabs((L-x)/vx_at_entrance);
		if(dt_for_last_step > 2*M_PI/(10.0*Omega)){
			dt_for_last_step = 2*M_PI/(10.0*Omega);
		}
		x = x_traj(dt_for_last_step, x_previous, vx_prime_previous, vy_previous, v_par_previous, slope);
		vx_prime = vx_prime_traj(dt_for_last_step, x_previous, vx_prime_previous, vy_previous, v_par_previous, slope);
	       	vy = vy_traj(dt_for_last_step, x_previous, vx_prime_previous, vy_previous, v_par_previous, slope);
		v_par = v_par_traj(dt_for_last_step, x_previous, vx_prime_previous, vy_previous, v_par_previous, slope);
		

		//runge_kutta(&x, &v_par, &vx_prime, &vy, &t, -dt_for_last_step, phi_arr, x_arr, 1);
		//printf("x = %.10f, v_par=%.10f, vx_prime = %.10f, vy = %.10f, v_par0 = %f, vx0 = %f\n", x, v_par, vx_prime, vy, -vx0*sin(alpha) + vz0*cos(alpha), vx0);
		//sleep(2);
	}
	//printf("x = %.10f, v_par=%.10f, vx_prime = %.10f, vy = %.10f, v_par0 = %f, vx0 = %f\n", x, v_par, vx_prime, vy, -vx0*sin(alpha) + vz0*cos(alpha), vx0);
	//sleep(2);
	double lambda = atan(sqrt(pow(vy,2.0)+pow(vx_prime, 2.0))/v_par);
	//printf("lambda = %f\n", 
	//double theta = arctan(-vy/vx_prime);
	double v = sqrt(pow(vy,2.0) + pow(vx_prime,2.0) + pow(v_par,2.0));
	//double f_inf(double v, double lambda)
	return f_inf(v, lambda);

}

void simulation(double x0, double v0, double lambda0, double theta0, double t0, double dtheta, double *position_arr, double *n_i, gsl_vector * rhs, int integral_flag, int current_index, int last_index, double* phi_arr, double *max_vz, double *min_vz, double *max_vx, double *min_vx, double *max_vy, double *min_vy, double dlambda_1, double dlambda_2){
	//printf("hey");
	double v=v0;
	double lambda=lambda0;
	double theta=theta0;
	double x=x0;
	double t=t0;
	double x_new = x0;
	double t_new = t0;
	double time_interval;
	//double t_arr[n+2], time_interval_arr[n+2], x_values_arr[n+2];
	int index;
	//#define BLOCK_SIZE 1000000
	//int *size_of_array_init = malloc(sizeof(double)*BLOCK_SIZE);
	//int max_index = BLOCK_SIZE - 1;
	double dt_for_step;
	//double position_place_holder[BLOCK_SIZE], time_arr_place_holder[BLOCK_SIZE], v_par_place_holder[BLOCK_SIZE], vx_prime_place_holder[BLOCK_SIZE], vy_place_holder[BLOCK_SIZE];
	double vx;
	int wall_flag = 1;
	double dlambda;
	double t_hit_wall;
	double max_vx_2 = v*sin(lambda)*cos(theta) + v*cos(lambda)*sin(theta);
	//if(wall_flag == 1){
	double vx_prime = v*sin(lambda)*cos(theta);
	double vy = -v*sin(lambda)*sin(theta);
	double v_par = v*cos(lambda);
	double x_previous;
	double t_previous;
	double vx_prime_previous;
	double v_par_previous;
	double vy_previous;
	double dv_for_integral;
	double v_squared_initial = vx_prime*vx_prime + vy*vy + v_par*v_par + phi_arr[n-1];
	vx = vx_prime*cos(alpha) - sin(alpha)*v_par;
	if(lambda0 < alpha){
		dlambda = dlambda_2;
		if(lambda0 == 0){
			dlambda_2/2.0;
		}
	}
	else if(lambda0 == alpha){
		dlambda = (dlambda_2 + dlambda_1)/2.0;
	}
	else{
		dlambda = dlambda_1;
	}
	dv_for_integral = dv;
	if(v0 == v_max){
		dv_for_integral = dv/2.0;
	}

	double a = -sin(lambda0)*v0*v0*v0*(cos(alpha)*cos(theta0)*sin(lambda0)-sin(alpha)*cos(lambda0))*f_inf(v0, lambda0)*dtheta*dv*dlambda;
	//printf("%f\n", x);
	//sleep(10);
	//
	int number_of_elements = n-1;
	int max_element = number_of_elements - 1;
	int i_element = max_element;
	double slope;
	double time_step = 0.0;
	double x_left, x_right;
	int i_element_previous;
	while(!((i_element == number_of_elements) || (i_element == -1))){
		i_element_previous = i_element;
		x_previous = x;
		vx_prime_previous = vx_prime; 
		vy_previous = vy;
		v_par_previous = v_par;
		x_right = position_arr[i_element+1];
		x_left = position_arr[i_element];
		slope = (phi_arr[i_element+1] - phi_arr[i_element])/(position_arr[i_element+1] - position_arr[i_element]);
		//printf("v0 = %f, theta0 = %f, lambda0 = %f\n", v0, theta0, lambda0);
		find_t_next(0.0, &x, vx_prime, vy, v_par, slope, &time_step, &i_element, position_arr);
		//if((i_element == number_of_elements) || (i_element == -1)){
		//	break;
		//}
		
		t += time_step;
		vx_prime = vx_prime_traj(time_step, x_previous, vx_prime_previous, vy_previous, v_par_previous, slope);
	       	vy = vy_traj(time_step, x_previous, vx_prime_previous, vy_previous, v_par_previous, slope);
		v_par = v_par_traj(time_step, x_previous, vx_prime_previous, vy_previous, v_par_previous, slope);
		//void calculate_contribution(double tau, double x0, double x1, double vx_prime1, double vy1, double v_par1, double slope, gsl_vector *rhs, double a, int i_elm, double *x_nodes)
		calculate_contribution(time_step, x_previous, x_left, x_right, vx_prime_previous, vy_previous, v_par_previous, slope, rhs, a, i_element_previous);
		//if(theta0 > 2.0){
		//	printf("i_elm = %d\n", i_element);
		//}
		//if((i_element == number_of_elements) || (i_element == -1)){
		//	break;
		//}
		//printf("%d\n", i_element);	
	}
	//printf("out");	



}
int main(){
	FILE *fout;
	//fout_2 = fopen("check_trajectories_when_code_breaks.txt", "w");
	//fout_3 = fopen("potential_when_code_breaks.txt", "w");
	double L;
	//L = 15.0;
	int n_D = 41;
	int n_mp = 61;
	n = n_D + n_mp;	
	#define BLOCK_SIZE 1000000
	//ng = 51;
	#define n_for_declaration 5000
	int m_v = 5;
	double gamma = 0.02;
	//int m_lambda = 20;
	int m_lambda_1 = 100;
	int m_lambda_2 = 100;
	int m_theta = 101;
	#define m_for_declaration 10
	#define m_v_for_declaration 200
	#define m_lambda_prime_for_declaration 100
	#define m_theta_for_declaration 100
	g_flag = 0;
	int n_for_plot = n;
	int integral_flag = 5;
	int n_g = 36;	
	//int m_lambda = 40;
	steps_per_cell = 20;
	double v_min_x = 0;
	v_max = 5.0;
	double alpha_deg = 5.0;
	double lambda_min = 0;
	lambda_max = M_PI/2.0;
	double theta_min = 0;
	theta_max = 2.0*M_PI;
	dv = fabs((double) (v_max)/(m_v));
	//dlambda = (double) (lambda_max-lambda_min)/(m_lambda);
	dtheta = (double) (theta_max-theta_min)/(m_theta-1);
	fout = fopen("potential_at_each_iteration_alpha_5_deg_with_Debye_sheath_delta_05_theta_1601_401_lambda_100_100_v_5_large_weight_for_iterations.txt", "w");	

	B = 1.0;
	alpha = alpha_deg*M_PI/180.0;//M_PI/100.0;//0.523599;//M_PI/100.0;
	printf("alpha = %f\n", alpha);
	//sleep(10);
	Omega = 1.0;
	e = 1;
	Z = 1;
	m_i = 1;
	double phi_0 = -2.8;
	double phi_inf = 0.0;
	T_e = 1.0;//0.001;

	p = 1; 
	double dlambda_1 = (lambda_max - alpha)/m_lambda_1;
	double dlambda_2 = alpha/m_lambda_2;
	int m_lambda = m_lambda_1 + m_lambda_2;
	
	//dx = (double) L/(n-1);
	//double x_arr[n_for_declaration], theta_arr[m_theta];
	double x_arr[n_for_declaration], v_arr[m_v], lambda_arr[m_lambda], theta_arr[m_theta], g_arr[n_g];
	//float phi_arr[n_for_declaration];
	
	
	
	for(int i=0; i<m_v; i++){
		v_arr[i] = (i+1)*dv;
		printf("%f\n", v_arr[i]);
	}
	//for(int i=0; i<m_theta; i++){
	//	theta_arr[i] = i*dtheta;
		//printf("%f\n", theta_arr[i]);
	//}
	for(int i=0; i<m_lambda_2; i++){
		lambda_arr[i] = i*dlambda_2;
	}
//	for(int i=0; i<m_lambda; i++){
//		lambda_arr[i] = i*dlambda;
	//	printf("%f\n", lambda_arr[i]);
	//}
	for(int i = 0; i<m_lambda_1; i++){
		lambda_arr[i+m_lambda_2] = alpha + i*dlambda_1;
	}

	for(int i = 0; i<m_lambda_1+m_lambda_2; i++){
		printf("lambda = %f\n", lambda_arr[i]);
	}
	//linspace(lambda_min, lambda_max, m_lambda, lambda_arr);
	//linspace(v_min_y, v_max_y, m_y, v_arr_y);
	//linspace(v_min_x, v_max_x, m_x, v_arr_x);
	for(int i = 0; i<n_D; i++){
		x_arr[i] = 20.0*gamma/((double) n_D-1)*i;	
		printf("x = %f\n", x_arr[i]);
	}
	kappa = 1.0;
	delta = 0.5;
	if(g_flag == 0){
		//kappa = 1.0;
		//delta = 0.4;
		for(int i=0; i<n_mp; i++){
		//v_arr[i] = (i+1)*dv;
		//printf("%d\n", i);
			x_arr[i+n_D] = pow(sqrt(kappa+(i+1)*delta)-sqrt(kappa), 2.0) + 20.0*gamma;
			printf("x=%f\n", x_arr[i+n_D]); 
		}
		L=x_arr[n-1];
	}
	else{
		L = pow(sqrt(kappa + (n-1)*delta) - sqrt(kappa), 2.0);
		linspace(0, sqrt(L), n_g, g_arr);
		dg = sqrt(L)/(n_g-1);
	}
	//sleep(10);
	for(int j =0; j<n_g; j++){
		printf("g = %f\n", g_arr[j]);
	}	
	if(g_flag == 1){
		n = n_g;
		n_for_plot = n_g;
	}
	//sleep(10);
	//dx = (x_arr[1] - x_arr[0])/(n-1);
	//double a;
	//a = 0;
	//for(i=0; i<n; i++){
	//	a += v_arr[i]*f_inf(v_arr[i])*dv; 
	//}	
	//normalization_factor=0;
	//for(int i = 0; i<m_v; i++){
	//	for(int l=0; l<m_lambda; l++){
			//printf("%f, %f, %f\n", v_arr_x[i], v_arr_y[l], v_arr_z[k]); 
	//		normalization_factor += 2*M_PI*f_inf(v_arr[i], lambda_arr[l])*v_arr[i]*v_arr[i]*sin(lambda_arr[l])*dv*dlambda;
	//	}
	//} 
	//printf("integral = %f\n", normalization_factor);
	double phi_arr[n+p-1];
	double a = phi_0 - phi_0/(1.0-pow((x_arr[n+p-2] + 1.0), 2.0));
	double b = phi_0 - a;
	for(int j =0; j<n+p-1; j++){
		//phi_arr[j] = 0.0;
		phi_arr[j] = a/(pow((x_arr[j] + 1), 2.0)) + b;
		/*if(g_arr[j] > -a/b){
			phi_arr[j] = 0.0;
		}
		else{
			phi_arr[j] = a + b*g_arr[j];
		}*/
	}
	/*double phi_arr[106] = {-2.97176500e+00, -2.81395383e+00, -2.63142392e+00, -2.46959100e+00,
	       -2.31326635e+00, -2.15919431e+00, -2.01111229e+00, -1.86719821e+00,
	       -1.72792296e+00, -1.59448570e+00, -1.46833809e+00, -1.35244222e+00,
	       -1.24288410e+00, -1.13969582e+00, -1.04281908e+00, -9.55112758e-01,
	       -8.73517907e-01, -7.97298986e-01, -7.26353848e-01, -6.62319127e-01,
	       -6.03685090e-01, -5.49274940e-01, -4.98963596e-01, -4.53630219e-01,
	       -4.12602649e-01, -3.74775886e-01, -3.40079258e-01, -3.09159557e-01,
	       -2.81319182e-01, -2.55738522e-01, -2.32335227e-01, -2.11427923e-01,
	       -1.92795263e-01, -1.75760161e-01, -1.60193648e-01, -1.46159124e-01,
	       -1.33567479e-01, -1.22001448e-01, -1.11398456e-01, -1.01896666e-01,
	       -9.33275984e-02, -8.54599502e-02, -7.82417869e-02, -7.16933300e-02,
	       -6.57116029e-02, -6.01712224e-02, -5.50519791e-02, -5.04667631e-02,
	       -4.62610517e-02, -4.23647696e-02, -3.87611926e-02, -3.55493464e-02,
	       -3.25868795e-02, -2.98435307e-02, -2.73101229e-02, -2.50737037e-02,
	       -2.29959727e-02, -2.10921281e-02, -1.94050634e-02, -1.79003501e-02,
	       -1.65161321e-02, -1.52458954e-02, -1.41005290e-02, -1.30739705e-02,
	       -1.21255285e-02, -1.12502868e-02, -1.04629729e-02, -9.74571536e-03,
	       -9.08037411e-03, -8.46383411e-03, -7.90922150e-03, -7.39717607e-03,
	       -6.92014946e-03, -6.47596111e-03, -6.07743928e-03, -5.70476346e-03,
	       -5.35631242e-03, -5.03310247e-03, -4.73882795e-03, -4.46267000e-03,
	       -4.20375790e-03, -3.96527598e-03, -3.74439164e-03, -3.53637332e-03,
	       -3.34075226e-03, -3.16114956e-03, -2.99338219e-03, -2.83505472e-03,
	       -2.68557458e-03, -2.54861495e-03, -2.41912754e-03, -2.29667768e-03,
	       -2.18170931e-03, -2.07476356e-03, -1.97377842e-03, -1.87831289e-03,
	       -1.78831770e-03, -1.70484640e-03, -1.62550006e-03, -1.54980347e-03,
	       -1.47888718e-03, -1.41210424e-03, -1.34877051e-03, -1.28883002e-03,
	       -1.23317097e-03, -1.17983960e-03};
	*/
	
	gsl_matrix *A = gsl_matrix_calloc(n+p-1, n+p-1);
	if(g_flag == 0){
		construct_matrix(n-1, x_arr, A, 0);
	}
	else{
		construct_matrix(n-1, g_arr, A, 1);
	}
	//printf("A01 = %f\n", gsl_matrix_get(A,1,0));
	int signum = 0;
	gsl_permutation *perm = gsl_permutation_alloc(n+p-1);
	gsl_linalg_LU_decomp(A, perm, &signum);
	//double input_arr[n];
	/*
	gsl_matrix *A_2 = gsl_matrix_calloc(n+p-1, n+p-1);
	if(g_flag == 0){
		construct_matrix(n-1, x_arr, A_2, 0);
	}
	else{
		construct_matrix(n-1, g_arr, A_2, 0);
	}
	int signum_2 = 0;
	gsl_permutation *perm_2 = gsl_permutation_alloc(n+p-1);
	gsl_linalg_LU_decomp(A_2, perm_2, &signum_2);
	//printf("A01 = %f\n", gsl_matrix_get(A,1,0));
	*/
	double max_vz, min_vz, max_vx, min_vx, max_vy, min_vy;
	double mean_change = 100000.0;
	for(int q=0; q<21; q++){
		max_vz = -100.0, min_vz = 100.0, max_vx = -100.0, min_vx = 0.0, max_vy = 0, min_vy = 100.0;
		double x, v, lambda, theta, dt, ind;
		double n_i[n_for_plot];
		for(int j=0; j<n_for_plot; j++){
			n_i[j] = 0.0;
		}
		dt = 0.001;

		clock_t start, end;
		double cpu_time_used;
		int m_theta_1, m_theta_2;

		gsl_vector *rhs = gsl_vector_calloc(n+p-1);
		//gsl_matrix *A = gsl_matrix_calloc(n+p-1, n+p-1);
		gsl_vector *density_values = gsl_vector_calloc(n+p-1);
	
		//construct_matrix(n-1, x_arr, A);

		start = omp_get_wtime();
		//for(int j=0; j<m_v; j++){
		//	for(int l=0; l<m_lambda; l++){
		//for(int j=0; j<m_v; j++){
		//for(int l=0; l<m_lambda; l++){
		m_theta_1 = 1601; 
		m_theta_2 = 401;
		double *initial_conditions_arr = malloc(((int) (m_lambda)*m_v*(m_theta_1 + m_theta_2)*4)*sizeof(double));

		int index = 0;
		for(int l=0; l<m_lambda;l++){
			printf("lambda = %f\n", lambda_arr[m_lambda -1-l]);
			for(int j=0; j<m_v; j++){
				v = v_arr[j];
				lambda = lambda_arr[m_lambda -1-l]; 
				if(lambda<alpha){
					theta_min = 0;
				}
				else{
					theta_min = acos(tan(alpha)/tan(lambda));
				}
				theta_max = 2*M_PI - theta_min;
				//dtheta = (theta_max - theta_min)/(m_theta-1);
	
				//for(int i = 0; i<m_theta; i++){
				//	theta_arr[i] = theta_min + i*dtheta;
				//}
				double epsilon = 1e-9;
				int gaussian_index;
				double theta_root = (theta_max+theta_min)/2.0;
				//printf("theta_min = %f, theta_max = %f\n", theta_min/M_PI, theta_max/M_PI);
				root_finder(&theta_root, lambda, theta_min, theta_max);
				//printf("theta_root = %f\n", theta_root);
				//m_theta_1 = 801;
				//double theta_arr_1[m_theta_1];
				dtheta = (theta_root - theta_min)/(m_theta_1-1);
				for(int i = 0; i<m_theta_1-1; i++){
					initial_conditions_arr[index] = lambda;
					initial_conditions_arr[index+1] = v;
					initial_conditions_arr[index+2] = theta_min + dtheta/2.0 + i*dtheta;
					initial_conditions_arr[index+3] = dtheta;
					index += 4;
					//theta_arr_1[i] = theta_min + dtheta/2.0 + i*dtheta;
					//printf("theta = %f\n", theta);
				}
				//for(int k=0; k<m_theta_1-1; k++){
					//printf("%d\n", j);
					//v = 1.0;//v_arr[j];
					//lambda = 0.9*M_PI/2.0; //lambda_arr[l];
				//	theta = theta_arr_1[k];
					//printf("theta = %f\n", theta);
				//	simulation(L, v, lambda, theta, 0.0, dt, x_arr, n_i, rhs, integral_flag, k, m_theta_1, phi_arr);
				//}

				if(lambda>alpha){
					//m_theta_2 = 101;
					//double epsilon = 1e-9;
					//double theta_arr_2[m_theta_2];
					dtheta = (theta_max - theta_root)/(m_theta_2-1);
					for(int i = 0; i<m_theta_2-1; i++){
						initial_conditions_arr[index] = lambda;
						initial_conditions_arr[index+1] = v;
						initial_conditions_arr[index+2] = theta_root + dtheta/2.0 + i*dtheta;
						initial_conditions_arr[index+3] = dtheta;
						index += 4;
						//theta_arr_2[i] = theta_root + dtheta/2.0 + i*dtheta;
					}
	
					//for(int k=0; k<m_theta_2-1;k++){
					//	theta = theta_arr_2[k];
					//	simulation(L, v, lambda, theta, 0.0, dt, x_arr, n_i, rhs, integral_flag, k, m_theta_2, phi_arr);
					//}
				}
			}
		}
		//for(int j =0; j<n+p-1; j++){
		//	printf("phi_arr = %f\n", phi_arr[j]);
		//}
		//printf("before realloc = %p, length = %ld\n", &initial_conditions_arr);
		//free(initial_conditions_arr);
		initial_conditions_arr = realloc(initial_conditions_arr, (index+1)*sizeof(double));
		//printf("after realloc = %p, length = %ld\n", &initial_conditions_arr);
		#pragma omp parallel shared(rhs) firstprivate(L, x_arr, n_i, integral_flag, phi_arr) reduction(max:max_vz) reduction(max:max_vx) reduction(max:max_vy) reduction(min:min_vz) reduction(min:min_vx) reduction(min:min_vy) //num_threads(1)
		{
					
			gsl_vector * rhs_i = gsl_vector_calloc(n+p-1);
			#pragma omp for schedule(dynamic)
			for (int i = 0; i<index; i+=4){
				lambda = initial_conditions_arr[i];
				v = initial_conditions_arr[i+1];
				theta = initial_conditions_arr[i+2];
				dtheta = initial_conditions_arr[i+3];
				//printf("lambda = %f, v = %f, theta = %f\n", lambda, v, theta);
				if(g_flag == 0){
					simulation(L, v, lambda, theta, 0.0, dtheta, x_arr, n_i, rhs_i, integral_flag, 0, 0, phi_arr, &max_vz, &min_vz, &max_vx, &min_vx, &max_vy, &min_vy, dlambda_1, dlambda_2);
				}
				else{
					simulation(L, v, lambda, theta, 0.0, dtheta, g_arr, n_i, rhs_i, integral_flag, 0, 0, phi_arr, &max_vz, &min_vz, &max_vx, &min_vx, &max_vy, &min_vy, dlambda_1, dlambda_2);
				}
				//printf("%d, %d\n", i, index);
			}
			printf("done\n");
			#pragma omp critical
			{
				gsl_vector_add(rhs, rhs_i);
			}
			//printf("done 2\n");
			gsl_vector_free(rhs_i);
		}
		free(initial_conditions_arr);
		printf("done parallel\n");
		//gsl_vector_scale(rhs, a);
		double rhs_for_plot[n+p-1];
		
		//int signum = 0;
		//gsl_permutation *perm = gsl_permutation_alloc(n+p-1);
		//gsl_linalg_LU_decomp(A, perm, &signum);
		//double input_arr[n];
	
		gsl_matrix *A_2 = gsl_matrix_calloc(n+p-1, n+p-1);
		if(g_flag == 0){
			construct_matrix_D(n-1, x_arr, A_2, phi_arr, gamma);
		}
		else{
			construct_matrix_D(n-1, g_arr, A_2, phi_arr, gamma);
		}
		//gsl_matrix_scale(D, gamma*gamma);


		//gsl_matrix *A_2 = gsl_matrix_calloc(n+p-1, n+p-1);
		//gsl_matrix_add(A_2, D);


		//int signum_2 = 0;
		//gsl_permutation *perm_2 = gsl_permutation_alloc(n+p-1);
		//gsl_linalg_LU_decomp(A_2, perm_2, &signum_2);

		for(int j=0; j<n+p-1; j++){
				rhs_for_plot[j] = gsl_vector_get(rhs, j);	
		}	
		//printf("check: %f\n", rhs_for_plot[0]);
		double density_values_arr[n+p-1];
		//solve_matrix(n, density_values, rhs, A);
		gsl_linalg_LU_solve(A, perm, rhs, density_values);
		for(int j = 0; j<n+p-1; j++){
			density_values_arr[j] = gsl_vector_get(density_values, j);
			printf("%f\n", gsl_vector_get(rhs, j));
		}
		gsl_vector_free(rhs);
		//gsl_matrix_free(A);
		//printf("check4: %f\n", rhs_for_plot[0]);
		//if(p==2){
		/*
		double h[p+1], h_x[p+1];
		double length = dx; 
	
		double x_arr_for_plot[n_for_plot];
		ouble dx_for_plot;
		int i_elm_prev; 
	
		dx_for_plot = (double) L/(n_for_plot-1);
	
		for(int l=0; l<n_for_plot; l++){
			x_arr_for_plot[l] = dx_for_plot*l;
		}
		i_elm_prev = 0;
	
		for(int j = 0; j<n_for_plot-1; j++){
			for(int i_elm_plot = i_elm_prev; i_elm_plot<n-1; i_elm_plot++){
				if((x_arr_for_plot[j] < x_arr[i_elm_plot+1]) && (x_arr_for_plot[j] >= x_arr[i_elm_plot])){
					//printf("yes\n");
					//printf("%d\n", i_elm_plot);
					i_elm_prev = i_elm_plot;
					break;
				}
			}
			printf("x = %f, i_elm = %d, x_left = %f, x_right = %f\n", x_arr_for_plot[j], i_elm_prev, x_arr[i_elm_prev], x_arr[i_elm_prev+1]);
			linear_element(x_arr, i_elm_prev, x_arr_for_plot[j], h, h_x, length);
			for(int l = 0; l<p+1; l++){
				//printf("h0 = %f, h1 = %f, h2 = %f\n", h[0], h[1], h[2]);
				//printf("%f\n", x_arr[j]);
				//printf("%d\n", j);
				n_i[j] += gsl_vector_get(density_values, i_elm_prev+l)*h[l];
			}
		}
		//printf("check5: %f\n", rhs_for_plot[0]);
		linear_element(x_arr, n-2, x_arr_for_plot[n_for_plot-1], h, h_x, length);
		for(int l=0; l<p+1; l++){
			//printf("h=%f\n", h[l]);
			n_i[n_for_plot-1] += gsl_vector_get(density_values, n-2+l)*h[l];
		}
		*/
		//printf("check3: %f\n", rhs_for_plot[0]);
		//}
	
		//else if(p==1){
		//	for(int j = 0; j<n; j++){
		//		n_i[j] += gsl_vector_get(density_values,j);
		//	}
		//}
		double phi_arr_new[n+p-1];
		//quasi_flag=0;
		
		printf("phi dens\n");
	 	double weight_for_iteration = 0.5;	
		gsl_vector *rhs_2 = gsl_vector_calloc(n+p-1);
		gsl_vector *phi_next = gsl_vector_calloc(n+p-1);
		if(g_flag == 0){
			make_vector_2(n-1, rhs_2, phi_arr, density_values_arr, x_arr);
		}
		else{
		 	make_vector_2(n-1, rhs_2, phi_arr, density_values_arr, g_arr);
		}
		
		set_boundary_conditions(rhs_2, A_2, phi_0, phi_inf);

		int signum_2 = 0;
		gsl_permutation *perm_2 = gsl_permutation_alloc(n+p-1);
		gsl_linalg_LU_decomp(A_2, perm_2, &signum_2);

		//solve_matrix(n, phi_next, rhs_2, A);
		gsl_linalg_LU_solve(A_2, perm_2, rhs_2, phi_next);
		//printf("A_00= %f\n", gsl_matrix_get(A, 0, 0));
		for(int l=0; l<n+p-1; l++){
			//printf("phi_arr before = %f\n", phi_arr[l]);
			//printf("dens - exp(phi) = %f\n", gsl_vector_get(density_values, l) - exp(phi_arr[l]));
			//double x_for_phi =(double) l*dx;
			printf("phi_next = %f, rhs_2 = %f\n", gsl_vector_get(phi_next, l), gsl_vector_get(rhs_2, l));
			phi_arr_new[l] = gsl_vector_get(phi_next, l);
			//phi_arr_new[l] = phi_arr[l] + exp(-phi_arr[l])*(gsl_vector_get(density_values, l) - exp(phi_arr[l]));
			printf("%f %f %f\n", phi_arr[l], gsl_vector_get(density_values,l), phi_arr_new[l]);
			fprintf(fout, "%2.10f %2.10f\n", phi_arr[l], gsl_vector_get(density_values,l));
			mean_change += fabs(exp(phi_arr[l]) - gsl_vector_get(density_values,l))/(exp(phi_arr[l]));
			phi_arr[l] = phi_arr_new[l]*weight_for_iteration + phi_arr[l]*(1.0-weight_for_iteration);
			printf("%2.10f,\n", phi_arr[l]);
		}
		//sleep(10);
		mean_change = mean_change/(n+p-1);
		gsl_vector_free(density_values);
		gsl_vector_free(rhs_2);
		gsl_vector_free(phi_next);	
	
		end = omp_get_wtime();
	
		cpu_time_used = ((double) (end - start));
	
		printf("time = %f\n", cpu_time_used);
		//printf("check2 = %f\n", rhs_for_plot[0]);	
		printf("p: %d, n: %d\n", p, n);
		printf("mean_change = %f\n", mean_change);
		
		/*fout = fopen("Solution_for_check_PIC.csv", "w");
 		
		//for(int i=0; i<n+p-1; i++){
		//	printf("%f\n", rhs_for_plot[i]);
		//}
	
		for(int i=0; i<n_for_plot; i++){
			printf("%5.10f\n", n_i[i]);
			fprintf(fout, "%5.15f\n", n_i[i]);
		}
		
		double sum_value=0;
		*/
		//for(int i = 0; i<m_v; i++){
		//	for(int l=0; l<m_lambda; l++){
		//		for(int k=0; k<m_theta; k++){
					//printf("f = %f\n", v_arr_par[i]);
		//			sum_value += fabs(v*v*sin(lambda)*f_inf(v_arr[i], lambda_arr[l])*dv*dlambda*dtheta);
		//		}
		//	}
		//}
		//printf("%1.14f, dv = %f, f(dv) = %f\n", sum_value, dv, f_inf(v_arr[0], lambda_arr[0]));
		//fclose(fout);
	//printf("%f\n", 10.4%3.0);
	}
	fclose(fout);
	
	FILE *fout_2;
	fout_2 = fopen("distribution_function_with_D_sheath_alpha_50_deg.txt", "w");

	gsl_matrix_free(A);
	gsl_permutation_free(perm);
	
	int m_v_dist = 200;
	int m_phi_dist = 400;
	int m_theta_dist = 200;
	double v_array[m_v_dist], phi_array[m_phi_dist], theta_array[m_theta_dist];
	//double distribution_function[m_vz][m_vx][m_vy];
	double x_dist = 0.0;
	linspace(0.0, 5.0, m_v_dist, v_array);
	linspace(0.0, 2.0*M_PI, m_phi_dist, phi_array);
	linspace(0.0, 0.5*M_PI, m_theta_dist, theta_array);
//calculate_f(double x0, double v_par0, double vx_prime0, double vy0, double t0, double dt, double *x_arr, double *phi_arr)
	double f_value;
	for(int i = 0; i<m_v_dist; i++){
		for(int j = 0; j<m_phi_dist; j++){
			for(int k = 0; k<m_theta_dist; k++){
				//if(g_flag == 0){
				//	distribution_function[i][j][k] = calculate_f(0, vz_array[i], vx_array[j], vy_array[k], 0, -0.01, x_arr, phi_arr, L);
				//}
				//else{
				//	distribution_function[i][j][k] = calculate_f(0, vz_array[i], vx_array[j], vy_array[k], 0, -0.01, g_arr, phi_arr, L);
				//}
				f_value = calculate_f(x_dist, v_array[i], phi_array[j], theta_array[k], 0, -0.01, x_arr, phi_arr, L);
				fprintf(fout_2, "%f %f %f %1.10f\n", v_array[i], phi_array[j], theta_array[k], f_value);
			}
		}
	}
	

	fclose(fout_2);
	
	/*	
	//int n_for_plot = 50;
	double phi_for_plot[n_for_plot];
	fout = fopen("Solution_phi_for_check_first.csv", "w");
	double h[p+1], h_x[p+1];
	double length = dx; 
	double x_arr_for_plot[n_for_plot];
	double dx_for_plot;
	int i_elm_prev; 
	dx_for_plot = (double) L/(n_for_plot-1);
	for(int l=0; l<n_for_plot; l++){
		x_arr_for_plot[l] = dx_for_plot*l;
	}
	i_elm_prev = 0;
	
	for(int j = 0; j<n_for_plot-1; j++){
		for(int i_elm_plot = i_elm_prev; i_elm_plot<n-1; i_elm_plot++){
			if((x_arr_for_plot[j] < x_arr[i_elm_plot+1]) && (x_arr_for_plot[j] >= x_arr[i_elm_plot])){
				//printf("yes\n");
				//printf("%d\n", i_elm_plot);
				i_elm_prev = i_elm_plot;
				break;
			}
		}
		//printf("x = %f, i_elm = %d, x_left = %f, x_right = %f\n", x_arr_for_plot[j], i_elm_prev, x_arr[i_elm_prev], x_arr[i_elm_prev+1]);
		linear_element(x_arr, i_elm_prev, x_arr_for_plot[j], h, h_x, length);
		for(int l = 0; l<p+1; l++){
			//printf("h0 = %f, h1 = %f, h2 = %f\n", h[0], h[1], h[2]);
			//printf("%f\n", x_arr[j]);
			//printf("%d\n", j);
			phi_for_plot[j] += phi_arr[i_elm_prev+l]*h[l];
		}
	}
	//printf("check5: %f\n", rhs_for_plot[0]);
	linear_element(x_arr, n-2, x_arr_for_plot[n_for_plot-1], h, h_x, length);
	for(int l=0; l<p+1; l++){
		//printf("h=%f\n", h[l]);
		phi_for_plot[n_for_plot-1] += phi_arr[n-2+l]*h[l];
	}
	for(int i=0; i<n_for_plot; i++){
		printf("%5.10f\n", phi_for_plot[i]);
		//fprintf(fout, "%5.15f\n", phi_for_plot[i]);
	}
	*/
	return 0;
}
