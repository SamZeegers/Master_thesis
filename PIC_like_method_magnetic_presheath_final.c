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
double phi_0; //value of the potential at the wall
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
double phi(double x, double *phi_arr, double *x_nodes){
	int j;
	double length;
	double h[p+1], h_x[p+1];
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

	linear_element(x_nodes, j, x, h, h_x, length, 0);
	double sum = 0;
	for(int l=0; l<p+1; l++){
		sum += h[l]*phi_arr[j+l];
	}
	return sum;//3.0/(pow(10.0,5.0))*pow(x-10.0, 5.0);
}

double evaluate_density(double x, double *n_arr, double *x_nodes){
	int j;
	double length; 
	double h[p+1], h_x[p+1];
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
	
	linear_element(x_nodes, j, x, h, h_x, length, 0);
	double sum=0;
	for(int l=0; l<p+1; l++){
		sum += h[l]*n_arr[j+l];
	}
	return sum; 
}

//definition of the gradient of the potential given the potential above
double phi_prime(double x, double *phi_arr, double *x_nodes, int squared_flag){
	int j;
	double length;
	double h[p+1], h_x[p+1];
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

//dx/dt the equation for the particle trajectory.
double dxdt(double t, double v_par, double vx_prime, double vy, double x){
	return cos(alpha)*vx_prime - sin(alpha)*v_par;
}
//dv/dt the equation for the particle trajectory.
double dvx_primedt(double t, double v_par, double vx_prime, double vy, double x, double phi_prime_value){
	return -Omega/B*cos(alpha)*phi_prime_value - Omega*vy;
}
double dvydt(double t, double v_par, double vx_prime, double vy, double x){
	return Omega*vx_prime;
}

double dv_pardt(double t, double v_par, double vx_prime, double vy, double x, double phi_prime_value){
	return Omega/B*sin(alpha)*phi_prime_value;
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
		
		//printf("dens = %f, phi = %f, x1 = %f\n", evaluate_density(x_1, n_arr, x_nodes), phi(x_1, phi_arr, x_nodes), x_1);
		//printf("dens = %f, phi = %f, x2 = %f\n", evaluate_density(x_2, n_arr, x_nodes), phi(x_2, phi_arr, x_nodes), x_2);
		//printf("dens = %f, phi = %f, x3 = %f\n", evaluate_density(x_3, n_arr, x_nodes), phi(x_3, phi_arr, x_nodes), x_3);
		//printf("dens = %f, phi = %f, x4 = %f\n", evaluate_density(x_4, n_arr, x_nodes), phi(x_4, phi_arr, x_nodes), x_4);
		//printf("dens = %f, phi = %f, x5 = %f\n", evaluate_density(x_5, n_arr, x_nodes), phi(x_5, phi_arr, x_nodes), x_5);

		ev_1 = phi(x_1, phi_arr, x_nodes) + exp(-phi(x_1, phi_arr, x_nodes))*(evaluate_density(x_1, n_arr, x_nodes) - exp(phi(x_1, phi_arr, x_nodes)));
		ev_2 = phi(x_2, phi_arr, x_nodes) + exp(-phi(x_2, phi_arr, x_nodes))*(evaluate_density(x_2, n_arr, x_nodes) - exp(phi(x_2, phi_arr, x_nodes)));
		ev_3 = phi(x_3, phi_arr, x_nodes) + exp(-phi(x_3, phi_arr, x_nodes))*(evaluate_density(x_3, n_arr, x_nodes) - exp(phi(x_3, phi_arr, x_nodes)));
		ev_4 = phi(x_4, phi_arr, x_nodes) + exp(-phi(x_4, phi_arr, x_nodes))*(evaluate_density(x_4, n_arr, x_nodes) - exp(phi(x_4, phi_arr, x_nodes)));
		ev_5 = phi(x_5, phi_arr, x_nodes) + exp(-phi(x_5, phi_arr, x_nodes))*(evaluate_density(x_5, n_arr, x_nodes) - exp(phi(x_5, phi_arr, x_nodes)));

		//printf("1 = %f, 2=%f, 3=%f, 4=%f, 5=%f\n", ev_1, ev_2, ev_3, ev_4, ev_5);	
		for(int k=0; k<p+1; k++){
			rhs_value = gsl_vector_get(rhs_2, i_elm+k);
			rhs_value += length/2.0*(128.0/225.0*ev_1*h_1[k] + (322.0 + 13.0*sqrt(70.0))/900.0*ev_2*h_2[k] + (322.0+13.0*sqrt(70.0))/900.0*ev_3*h_3[k] + (322.0-13.0*sqrt(70.0))/900.0*ev_4*h_4[k] + (322.0-13.0*sqrt(70.0))/900.0*ev_5*h_5[k]);
			//rhs_value += length/2.0*(128.0/225.0*ev_1*h_1[k]*2.0*x_1 + (322.0 + 13.0*sqrt(70.0))/900.0*ev_2*h_2[k]*2.0*x_2 + (322.0+13.0*sqrt(70.0))/900.0*ev_3*h_3[k]*2.0*x_3 + (322.0-13.0*sqrt(70.0))/900.0*ev_4*h_4[k]*2.0*x_4 + (322.0-13.0*sqrt(70.0))/900.0*ev_5*h_5[k]*2.0*x_5);
			gsl_vector_set(rhs_2, i_elm+k, rhs_value);
		}


	}
}


void runge_kutta(double *x, double *v_par, double *vx_prime, double *vy, double *t, double dt, double *phi_arr, double *position_nodes, int quadratic_flag){
	double k1,k2,k3,k4,l1,l2,l3,l4,m1,m2,m3,m4,p1,p2,p3,p4;
	double phi_prime_value;	
	//double vx_prime = *v*sin(*lambda)*cos(*theta);
	//double vy = *v*sin(*lambda)*sin(*theta);
	//double v_par = *v*cos(*lambda);
	//printf("FIRST : x = %f, v = %f, lambda = %f, theta = %f, v_par = %f, vx_prime = %f, vy = %f, vx = %f\n", *x, *v, *lambda, *theta, *v_par, *vx_prime, *vy, vx_prime*cos(alpha)-v_par*sin(alpha));
	if(g_flag == 0){
		phi_prime_value = phi_prime(*x, phi_arr, position_nodes, 0);
	}
	else{
		if (*x <= 0.0){
			phi_prime_value = phi_prime(0.0, phi_arr, position_nodes, 0)*1.0/(2.0*sqrt(1e-8));
		}
		else{
			phi_prime_value = phi_prime(sqrt(*x), phi_arr, position_nodes, 0)*1.0/(2.0*sqrt(*x));
		}
	}
	if(quadratic_flag == 1){
		phi_prime_value = phi_prime(*x, phi_arr, position_nodes,1);
	}
	k1 = dt*dxdt(*t, *v_par, *vx_prime, *vy, *x);
	//printf("%f\n", x+k1);
	l1 = dt*dv_pardt(*t, *v_par, *vx_prime, *vy, *x, phi_prime_value);
	m1 = dt*dvx_primedt(*t, *v_par, *vx_prime, *vy, *x, phi_prime_value);
	p1 = dt*dvydt(*t, *v_par, *vx_prime, *vy, *x);
		
	if(g_flag == 0){
		phi_prime_value = phi_prime(*x + 0.5*k1, phi_arr, position_nodes, 0);
	}
	else{
		if (*x + 0.5*k1 <= 0.0) {
			phi_prime_value = phi_prime(0.0, phi_arr, position_nodes, 0)*1.0/(2.0*sqrt(1e-8));
		}
		else{
			phi_prime_value = phi_prime(sqrt(*x + 0.5*k1), phi_arr, position_nodes, 0)*1.0/(2.0*sqrt(*x+0.5*k1));
		}
	}
	if(quadratic_flag == 1){
		phi_prime_value = phi_prime(*x + 0.5*k1, phi_arr, position_nodes,1);
	}
	//printf("%f\n", v + 0.5*l1);
	k2 = dt*dxdt(*t + 0.5*dt, *v_par + 0.5*l1, *vx_prime + 0.5*m1, *vy + 0.5*p1, *x + 0.5*k1);
	//printf("%f\n", x + k2);
	l2 = dt*dv_pardt(*t + 0.5*dt, *v_par + 0.5*l1, *vx_prime + 0.5*m1, *vy + 0.5*p1, *x + 0.5*k1, phi_prime_value);
	m2 = dt*dvx_primedt(*t + 0.5*dt, *v_par + 0.5*l1, *vx_prime + 0.5*m1, *vy + 0.5*p1, *x + 0.5*k1, phi_prime_value);
	p2 = dt*dvydt(*t + 0.5*dt, *v_par + 0.5*l1, *vx_prime + 0.5*m1, *vy + 0.5*p1, *x + 0.5*k1);

	if(g_flag == 0){
		phi_prime_value = phi_prime(*x + 0.5*k2, phi_arr, position_nodes, 0);
	}
	else{
		if(*x + 0.5*k2 <= 0.0){
			phi_prime_value = phi_prime(0.0, phi_arr, position_nodes, 0)*1.0/(2.0*sqrt(1e-8));
		}
		else{
			phi_prime_value = phi_prime(sqrt(*x + 0.5*k2), phi_arr, position_nodes, 0)*1.0/(2.0*sqrt(*x + 0.5*k2));
		}
	}
	if(quadratic_flag == 1){
		phi_prime_value = phi_prime(*x+0.5*k2, phi_arr, position_nodes, 1);
	}
	k3 = dt*dxdt(*t + 0.5*dt, *v_par + 0.5*l2, *vx_prime + 0.5*m2, *vy + 0.5*p2, *x + 0.5*k2);
	l3 = dt*dv_pardt(*t + 0.5*dt, *v_par + 0.5*l2, *vx_prime + 0.5*m2, *vy + 0.5*p2, *x + 0.5*k2, phi_prime_value);
	m3 = dt*dvx_primedt(*t + 0.5*dt, *v_par + 0.5*l2, *vx_prime + 0.5*m2, *vy + 0.5*p2, *x + 0.5*k2, phi_prime_value);
	p3 = dt*dvydt(*t + 0.5*dt, *v_par + 0.5*l2, *vx_prime + 0.5*m2, *vy + 0.5*p2, *x + 0.5*k2);
	
	if(g_flag == 0){
		phi_prime_value = phi_prime(*x + k3, phi_arr, position_nodes, 0);
	}
	else{
		if(*x + k3 <= 0.0){
			phi_prime_value = phi_prime(0.0, phi_arr, position_nodes, 0)*1.0/(2.0*sqrt(1e-8));
		}
		else{
			phi_prime_value = phi_prime(sqrt(*x + k3), phi_arr, position_nodes, 0)*1.0/(2.0*sqrt(*x + k3));
		}
	}
	if(quadratic_flag == 1){
		phi_prime_value = phi_prime(*x + k3, phi_arr, position_nodes, 1);
	}
	k4 = dt*dxdt(*t + dt, *v_par + l3, *vx_prime + m3, *vy + p3, *x + k3);
	l4 = dt*dv_pardt(*t + dt, *v_par + l3, *vx_prime + m3, *vy + p3, *x + k3, phi_prime_value);
	m4 = dt*dvx_primedt(*t + dt, *v_par + l3, *vx_prime + m3, *vy + p3, *x + k3, phi_prime_value);
	p4 = dt*dvydt(*t + dt, *v_par + l3, *vx_prime + m3, *vy + p3, *x + k3);
	
	*x = *x + 1.0/6.0*(k1 + 2.0*k2 + 2.0*k3 + k4);
	*v_par = *v_par + 1.0/6.0*(l1 + 2.0*l2 + 2.0*l3 + l4);
	*vx_prime = *vx_prime + 1.0/6.0*(m1 + 2.0*m2 + 2.0*m3 + m4);
	*vy = *vy + 1.0/6.0*(p1 + 2.0*p2 + 2.0*p3 + p4); 
	*t = *t + dt;
	
	
	/*
	*t = *t + dt;
	*v = *v;
	*lambda = *lambda;
	*theta = *theta + Omega*dt;
	*x = *x + *v*sin(*lambda)*cos(alpha)*(sin(*theta) - sin(*theta - Omega*dt)) - *v*sin(alpha)*cos(*lambda)*dt*Omega;
	*/		
}
void calculate_contribution(double x, double x_prev, double dt, gsl_vector *rhs, double a, int i_elm, int i_elm_prev, double *x_nodes){
	double h_1[p+1], h_1x[p+1], h_2[p+1], h_2x[p+1];
	double added_value_to_integral, added_value_to_integral_prev;
	double integral_value, integral_value_prev;	
	double length;
	//printf("%d, %d\n", i_elm, i_elm_prev);
	if(g_flag == 0){
		linear_element(x_nodes, i_elm, x, h_1, h_1x, length, 0);
		linear_element(x_nodes, i_elm_prev, x_prev, h_2, h_2x, length, 0);
	}
	else{
		linear_element(x_nodes, i_elm, sqrt(x), h_1, h_1x, length, 0);
		linear_element(x_nodes, i_elm_prev, sqrt(x_prev), h_2, h_2x, length, 0);
	}
	for(int l = 0; l<p+1; l++){
		integral_value = gsl_vector_get(rhs, i_elm+l);
		//integral_value_prev = gsl_vector_get(rhs, i_elm_prev+l);
		added_value_to_integral = a*(dt/2.0)*h_1[l];
		//added_value_to_integral_prev = a*(dt/2.0)*h_2[l];
		integral_value +=added_value_to_integral;
		//integral_value_prev += added_value_to_integral_prev;
		gsl_vector_set(rhs, i_elm+l, integral_value);	
		//gsl_vector_set(rhs, i_elm_prev+l, integral_value);	
	}
	for(int l = 0; l<p+1; l++){
		integral_value_prev = gsl_vector_get(rhs, i_elm_prev+l);
		added_value_to_integral_prev = a*(dt/2.0)*h_2[l];
		integral_value_prev += added_value_to_integral_prev;
		gsl_vector_set(rhs, i_elm_prev+l, integral_value_prev);
	}

}

double calculate_f(double x0, double vz0, double vx0, double vy0, double t0, double dt, double *x_arr, double *phi_arr, double L){
	double x=x0;
//	double Kx0 = phi(0.0, phi_arr, x_arr) - phi(sqrt(0.01*vx0*dt), phi_arr, x_arr) + 1.0/2.0*vx0*vx0 + vy0*Omega*vx0*0.01*dt*cos(alpha)*B;
//	if(Kx0 < 0){
//		return 0.0;
//	}
//	x = x0 + vx0*0.01*dt;
//	vx0 = -sqrt(2.0*Kx0);
	//x = x0 + vx0*dt;
	double vx_prime = vx0*cos(alpha) + vz0*sin(alpha);
	double v_par = -vx0*sin(alpha) + vz0*cos(alpha);
	vy0 = - vy0;
	double vy = vy0;
	double t=t0;
	double vx_at_entrance;
	double dt_for_last_step;
	//if(fabs(vx0) < 1){
	//	printf("x = %f, v_par = %f, vx_prime = %f, vy = %f, vx = %f\n", x, v_par, vx_prime, vy, vx_prime*cos(alpha) - v_par*sin(alpha));
	//	sleep(2);
	//}
	double slope_sqrt = (phi(x_arr[1], phi_arr, x_arr) - phi_arr[0])/sqrt(x_arr[1]);
	double x_max = slope_sqrt*slope_sqrt/(pow(2.0*vy0*cos(alpha)*B, 2.0));
	double energy_at_max = 0.5*vx0*vx0 - Omega/B*slope_sqrt*sqrt(x_max) + Omega*vy0*cos(alpha)*x_max;
	if(vy > 0){
		if(energy_at_max < 0){
			return 0.0;
		}
		else{
			x = x_arr[1];
			vx_prime = -sqrt(vx0*vx0 - 2.0*Omega/B*slope_sqrt*sqrt(x) + 2.0*Omega*vy0*cos(alpha)*x)*cos(alpha) + vz0*sin(alpha); 
		}
	}
	else{
		if(vx0*vx0 - 2.0*Omega/B*slope_sqrt*sqrt(x_arr[1]) + 2.0*Omega*vy0*cos(alpha)*x_arr[1] < 0){
			return 0.0;
		}
		else{
			x = x_arr[1];
			vx_prime = -sqrt(vx0*vx0 - 2.0*Omega/B*slope_sqrt*sqrt(x) + 2.0*Omega*vy0*cos(alpha)*x)*cos(alpha) + vz0*sin(alpha);
		}
	}
	while (x<L){
		runge_kutta(&x,&v_par,&vx_prime,&vy,&t, dt, phi_arr, x_arr,0);
		
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
	runge_kutta(&x, &v_par, &vx_prime, &vy, &t, -dt, phi_arr, x_arr, 1);
	while(x < L-1e-10){
		if(g_flag == 0){
			vx_at_entrance = 2.0*sqrt(1.0/2.0*(pow(vx_prime,2.0)+pow(v_par,2.0) + pow(vy,2.0)) - Z*e*phi(x, phi_arr, x_arr)/m_i);
		}
		else{
			vx_at_entrance = 2.0*sqrt(1.0/2.0*(pow(vx_prime,2.0)+pow(v_par,2.0) + pow(vy,2.0)) - Z*e*phi(sqrt(x), phi_arr, x_arr)/m_i);
		}
		dt_for_last_step = fabs((L-x)/vx_at_entrance);
		if(dt_for_last_step > 2*M_PI/(10.0*Omega)){
			dt_for_last_step = 2*M_PI/(10.0*Omega);
		}
		runge_kutta(&x, &v_par, &vx_prime, &vy, &t, -dt_for_last_step, phi_arr, x_arr, 0);
		//printf("x = %.10f, v_par=%.10f\n", x, v_par);
	}
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
	double vy = v*sin(lambda)*sin(theta);
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
	//printf("vx = %f\n", vx);
	if(vx > 0){
		//printf("vx = %f, vx is positive\n", vx);
		return;
	}
	int if_flag = 1; 
	//printf("%f\n", v_parallel);
	//position_place_holder = malloc(sizeof(double)*BLOCK_SIZE);
	//time_arr_place_holder = malloc(sizeof(double)*BLOCK_SIZE);
	//vx_prime_place_holder = malloc(sizeof(double)*BLOCK_SIZE);
	//v_par_place_holder = malloc(sizeof(double)*BLOCK_SIZE);
	//vy_place_holder = malloc(sizeof(double)*BLOCK_SIZE);

	//vy = 0.0;
	//vz = 0.0;
	//printf("position = %p\n", position_place_holder);
	//printf("time_arr = %p\n", time_arr_place_holder);
	int i_element;
	if(g_flag == 0){
		i_element = (int) ((pow(sqrt(x) + sqrt(kappa), 2.0)-kappa)/delta); 
	}
	else{
		i_element = (int) (sqrt(x)/dg);
	}
	if(i_element == n-1){
		i_element = n-2; 
	}
	if(g_flag ==0){
		dx = position_arr[i_element+1] - position_arr[i_element]; 
	}
	else{
		dx = pow(position_arr[i_element+1], 2.0) - pow(position_arr[i_element], 2.0);
	}
	double max_v;
	double v_squared;
	v_squared = pow(v_par, 2.0) + pow(vy, 2.0) + pow(vx_prime, 2.0);
	
	max_v = 2.0*sqrt(0.5*v_squared + phi(position_arr[i_element+1], phi_arr, position_arr) - phi(position_arr[i_element], phi_arr, position_arr));
	
	//v_at_wall = -2.0*sqrt(1.0/2.0*pow(v, 2.0) + Z*e*phi(x)/m_i - Z*e*phi_0/m_i);
	dt_for_step = fabs(dx/max_v)/steps_per_cell; //-x/v_at_wall;
	//printf("dt_for_step = %f, vx = %f, steps = %d\n ", dt_for_step, vx, steps_per_cell);
	//printf("dx = %f, max_v = %f, dt = %f\n", dx, max_v, dt_for_step);
	//sleep(10);

	//vx_prev = vx;
	if (dt_for_step > 2*M_PI/(Omega*10.0)){
		dt_for_step = 2*M_PI/(Omega*10.0);
	}

	int contribution_flag; 
	int particle_hit_wall=0;
	int particle_exited=0;
	int i_element_prev = i_element;
	int first_time_entering = -1;
	double vx_at_wall, dt_for_last_step, vx_prev;
	double phi_term;
	while(x > 0){
		
		//v_at_wall = -2.0*sqrt(1.0/2.0*pow(v, 2.0) + Z*e*phi(x)/m_i - Z*e*phi_0/m_i);
		//dt_for_step = fabs(dx/(v*sin(lambda)*cos(alpha) + v*cos(lambda)*sin(alpha)))/steps_per_cell; //-x/v_at_wall;
		//printf("dt_for_step = %f, vx = %f, steps = %d\n ", dt_for_step, vx, steps_per_cell);
		 

		vx_prev = vx;
		x_previous = x;
		vx_prime_previous = vx_prime;
		v_par_previous = v_par;
		vy_previous = vy;
		t_previous = t;


		runge_kutta(&x, &v_par, &vx_prime, &vy, &t, dt_for_step, phi_arr, position_arr, 0);
		
		
		if(v_par < 0){
			//free(time_arr_place_holder);
			//free(position_place_holder);
			//free(v_par_place_holder);
			//free(vx_prime_place_holder);
			//free(vy_place_holder);
			return; 
		}
		//printf("x = %f, vx = %f\n", x, vx); 
		i_element_prev = i_element;
		vx = vx_prime*cos(alpha) - v_par*sin(alpha);
		if(x!=x){
			printf("x = %f, term = %f, dt = %f, first_enter_index = %d, phi_term = %f, v_max = %f, i_elm = %d, dx = %f\n", x, fabs(dx/v_max)/steps_per_cell, dt_for_step, first_time_entering, phi_term, v_max, i_element, dx);
		}
		
		if((x<0) || (x>x0)){
			//printf("%f\n", x); 
			//hit_wall_or_entrance: 
			break;
		}



		if(g_flag == 0){
			i_element = (int) ((pow(sqrt(x) + sqrt(kappa), 2.0)-kappa)/delta);
		}
		else{
			i_element = (int) (sqrt(x)/dg);
			//printf("x = %f, sqrt(x) = %f, dg = %f, i_elm =%d, sqrt(x)/dg = %f\n", x, sqrt(x), dg, i_element, sqrt(x)/dg);
			//sleep(10);
		}
		//printf("x = %f, i_elm = %d, x_i = %f, x_i+1 = %f\n",x,  i_element, pow(position_arr[i_element], 2.0), pow(position_arr[i_element+1], 2.0));
		//sleep(0.5);	
		if(i_element == n-1){
			i_element = n-2; 
		}
			
		if(i_element < i_element_prev){
			first_time_entering = -1*first_time_entering;
			//printf("%d, i_elm = %d, i_elm_prev = %d\n", first_time_entering, i_element, i_element_prev);
		}
		contribution_flag = 1;
		if((first_time_entering == 1)&&(i_element < i_element_prev)){
			//printf("x_prev = %f, t_prev = %f, v_par_prev = %f, vx_prime_prev = %f, vy_prev = %f\n", x, t, v_par, vx_prime, vy);
			x = x_previous;
			t = t_previous;
			v_par = v_par_previous;
			vx_prime = vx_prime_previous;
			vy = vy_previous;
			//printf("x = %f, t = %f, v_par = %f, vx_prime = %f, vy = %f\n", x, t, v_par, vx_prime, vy);
			//dx = x_arr[i_element+1] - x_arr[i_element];
			//printf("%f\n", dx);
			i_element = i_element+1;
			contribution_flag = 0;
			//printf("i_element_after = %d\n", i_element);		
		}
		
		if(first_time_entering == 1){
			if(g_flag == 0){
				dx = position_arr[i_element] - position_arr[i_element-1];
			}
			else{
				dx = pow(position_arr[i_element], 2.0) - pow(position_arr[i_element-1], 2.0);
			}
			phi_term = phi(position_arr[i_element], phi_arr, position_arr) - phi(position_arr[i_element-1], phi_arr, position_arr);
			//printf("%f\n", dx);
		}
		else{
			if(g_flag ==0){
				dx = position_arr[i_element+1] - position_arr[i_element];
			}
			else{
				dx = pow(position_arr[i_element+1],2.0) - pow(position_arr[i_element], 2.0); 
			}
			phi_term = phi(position_arr[i_element+1], phi_arr, position_arr) - phi(position_arr[i_element], phi_arr, position_arr);
			//printf("phi = %f\n", phi_term);
		}
		//printf("%d\n", contribution_flag);
		if(contribution_flag == 1){
			calculate_contribution(x, x_previous, dt_for_step, rhs, a, i_element, i_element_prev, position_arr);
		}
		v_squared = pow(v_par, 2.0) + pow(vy, 2.0) + pow(vx_prime, 2.0);
		//v_at_wall = -2.0*sqrt(1.0/2.0*pow(v, 2.0) + Z*e*phi(x)/m_i - Z*e*phi_0/m_i);
		max_v = 2.0*sqrt(0.5*v_squared + phi_term);//Omega/B*(phi_arr[i_element+1]-phi_arr[i_element])/dx*t + max_vx;
		
		//printf("max v = %f, max_vx = %f\n", max_v, max_vx);
		//printf("v_squared = %f\n", v_squared);	
		dt_for_step = fabs(dx/max_v)/steps_per_cell; //-x/v_at_wall;
		if(i_element == 0){
			dt_for_step = dt_for_step/steps_per_cell;
		}
		
		//printf("dt_for_step = %f, vx = %f, steps = %d\n ", dt_for_step, vx, steps_per_cell);
		//printf("i_elm = %d\n", i_element);

		//vx_prev = vx;
		if (dt_for_step > 2*M_PI/(Omega*10.0)){
			//printf("yes\n");
			dt_for_step = 2*M_PI/(Omega*10.0);
		}
		if(dt_for_step != dt_for_step){
			//dt_for_step = fabs(dx/v_max)/steps_per_cell;
			printf("first term = %f, second term = %f\n",  phi(position_arr[i_element+1], phi_arr, position_arr),  phi(position_arr[i_element], phi_arr, position_arr));
			printf("phi_arr_1 = %f, phi_arr_2 = %f, phi_arr_3 = %f, phi_arr_4 = %f\n", phi_arr[i_element], phi_arr[i_element+1], phi_arr[i_element+2], phi_arr[i_element+3]);
			printf("x = %f, term = %f, dt = %f, first_enter_index = %d, phi_term = %f, v_max = %f, i_elm = %d, dx = %f\n", x, fabs(dx/max_v)/steps_per_cell, dt_for_step, first_time_entering, phi_term, max_v, i_element, dx);
		}

		
	}
	//printf("bulk done\n");
	//double t_hit_wall;
	if((x < 0)||(particle_hit_wall==1)){
		//printf("x < 0\n");
		x = x_previous;
		//printf("x=%f\n", x);
		//printf("%.15f\n", x);
		v_par = v_par_previous;
		vx_prime = vx_prime_previous;
		vy = vy_previous; 
		t = t_previous;
		//printf("x = %f, last_grid = %f\n", x, x_arr[1]);
		//printf("theta hit wall = %f\n", theta0); 
		while(x>1e-10){
			x_previous = x;
			vx_prime_previous = vx_prime;
			v_par_previous = v_par;
			vy_previous = vy;
			t_previous = t;
			v_squared = pow(v_par, 2.0) + pow(vy, 2.0) + pow(vx_prime, 2.0);
			if(g_flag == 0){
				max_v = 2.0*sqrt(0.5*v_squared + phi(x, phi_arr, position_arr) - phi(position_arr[0], phi_arr, position_arr));
			}
			else{
				max_v = 2.0*sqrt(0.5*v_squared + phi(sqrt(x), phi_arr, position_arr) - phi(position_arr[0], phi_arr, position_arr));
			}
			//vx_at_wall = -2.0*sqrt(1.0/2.0*(pow(vx_prime,2.0) + pow(vy,2.0) + pow(v_par,2.0)) + Z*e*phi(x)/m_i - Z*e*phi_0/m_i);
			//printf("vx = %f\n", vx_at_wall);
			dt_for_last_step = fabs(x/max_v);
			if(dt_for_last_step > 2*M_PI/(Omega*10.0)){
				dt_for_last_step = 2*M_PI/(Omega*10.0); 
			}
			//printf("x = %f\n", x);
			runge_kutta(&x, &v_par, &vx_prime, &vy, &t, dt_for_last_step, phi_arr, position_arr, 0);
			//printf("x = %f\n", x);
			//printf("%f, max_v = %f, v_squared = %f\n", x, max_v, v_squared);
			if(x<0){
				printf("%f, max_v = %f, v_squared = %f\n", x, max_v, v_squared);
			}
			calculate_contribution(x, x_previous, dt_for_last_step, rhs, a, 0, 0, position_arr);
			//vy = 0.0;
			//vz = 0.0;
			//if(vx_prime*cos(alpha) - v_par*sin(alpha) > 0){
			//	printf("x= %f, vx_prime = %f, vy = %f, v_par = %f\n", x, vx_prime, vy, v_par);
			//}
		}
		
		*max_vz = fmaxf(*max_vz, (vx_prime*sin(alpha) + cos(alpha)*v_par));
		*min_vz = fminf(*min_vz, (vx_prime*sin(alpha) + cos(alpha)*v_par));
		*max_vx = fmaxf(*max_vx, (vx_prime*cos(alpha) - v_par*sin(alpha)));
		*min_vx = fminf(*min_vx, (vx_prime*cos(alpha) - v_par*sin(alpha)));
		*max_vy = fmaxf(*max_vy, vy);
		*min_vy = fminf(*min_vy, vy);
		t_hit_wall = t;
		//printf("%f\n", t_hit_wall);
		
		
	}
	
	else if((x > x0)||(particle_exited == 1)){
		double vx_at_entrance;
		x = x_previous;
		v_par = v_par_previous;
		vx_prime = vx_prime_previous;
		vy = vy_previous;
		t = t_previous;
		//printf("x > 0\n");
		//printf("theta back to entrance= %f\n", theta0);	
		while(x < x0-1e-10){
			x_previous = x;
			if(g_flag == 0){
				vx_at_entrance = 2.0*sqrt(1.0/2.0*(pow(vx_prime,2.0)+pow(v_par,2.0) + pow(vy,2.0)) - Z*e*phi(sqrt(x), phi_arr, position_arr)/m_i);
			}
			else{
				 vx_at_entrance = 2.0*sqrt(1.0/2.0*(pow(vx_prime,2.0)+pow(v_par,2.0) + pow(vy,2.0)) - Z*e*phi(sqrt(x), phi_arr, position_arr)/m_i);
			}
			dt_for_last_step = fabs((x0-x)/vx_at_entrance);
			if(dt_for_last_step > 2*M_PI/(10.0*Omega)){
				dt_for_last_step = 2*M_PI/(10.0*Omega);
			}
			runge_kutta(&x, &v_par, &vx_prime, &vy, &t, dt_for_last_step, phi_arr, position_arr, 0);
			//printf("x = %.10f, x_previous = %.10f, L = %.10f, v=%.10f\n", x, x_previous, x0, v);
			calculate_contribution(x, x_previous, dt_for_last_step, rhs, a, n-2, n-2, position_arr);
		}	
	}	
}


int main(){
	//FILE *fout;
	//fout_2 = fopen("check_trajectories_when_code_breaks.txt", "w");
	//fout_3 = fopen("potential_when_code_breaks.txt", "w");
	double L;
	//L = 15.0;
	n = 31;
	#define BLOCK_SIZE 1000000
	//ng = 51;
	#define n_for_declaration 5000
	int m_v = 10;
	//int m_lambda = 20;
	int m_lambda_1 = 20;
	int m_lambda_2 = 30;
	int m_theta = 101;
	double alpha_deg = 50.0;
	#define m_for_declaration 10
	#define m_v_for_declaration 200
	#define m_lambda_prime_for_declaration 100
	#define m_theta_for_declaration 100
	g_flag = 0;
	int n_for_plot = 81;
	int integral_flag = 5;
	int n_g = 36;	
	//int m_lambda = 40;
	steps_per_cell = 20;
	double v_min_x = 0;
	v_max = 5.0;
	double lambda_min = 0;
	lambda_max = M_PI/2.0;
	double theta_min = 0;
	theta_max = 2.0*M_PI;
	dv = fabs((double) (v_max)/(m_v));
	//dlambda = (double) (lambda_max-lambda_min)/(m_lambda);
	dtheta = (double) (theta_max-theta_min)/(m_theta-1);
	//fout = fopen("potential_at_each_iteration_quasi_neutrality_alpha_60_deg.txt", "w");	


	B = 1.0;
	alpha = alpha_deg*M_PI/180.0;//M_PI/100.0;
	Omega = 1.0;
	e = 1;
	Z = 1;
	m_i = 1;
	phi_0 =0.0;// -3.0;
	T_e = 1.0;//0.001;


	p = 2; 
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
	kappa = 1.0;
	delta = 0.6;
	if(g_flag == 0){
		//kappa = 1.0;
		//delta = 0.4;
		for(int i=0; i<n; i++){
		//v_arr[i] = (i+1)*dv;
		//printf("%d\n", i);
			x_arr[i] = pow(sqrt(kappa+i*delta)-sqrt(kappa), 2.0);
			printf("x=%f\n", x_arr[i]); 
		}
		L=x_arr[n-1];
	}
	else{
		L = pow(sqrt(kappa + (n-1)*delta) - sqrt(kappa), 2.0);
		linspace(0, sqrt(L), n_g, g_arr);
		dg = sqrt(L)/(n_g-1);
	}
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
	double a = -0.20;
	double b = 0.15;
	for(int j =0; j<n+p-1; j++){
		phi_arr[j] = 0.0;
		/*if(g_arr[j] > -a/b){
			phi_arr[j] = 0.0;
		}
		else{
			phi_arr[j] = a + b*g_arr[j];
		}*/
	}
	
	//double phi_arr[32] = {-2.42674638e-01, -2.12832506e-01, -1.78011372e-01, -1.43671220e-01,
        //-1.16681075e-01, -9.30041225e-02, -7.42468181e-02, -6.02890844e-02,
        //-4.89904203e-02, -3.94841323e-02, -3.23430104e-02, -2.60677730e-02,
        //-2.18092175e-02, -1.79080894e-02, -1.45886697e-02, -1.24568257e-02,
        //-9.89972550e-03, -8.58126960e-03, -6.79900030e-03, -5.82729050e-03,
        //-4.69289100e-03, -3.84628810e-03, -3.17191960e-03, -2.44952710e-03,
        //-2.06623180e-03, -1.46110680e-03, -1.34648760e-03, -6.98826000e-04,
        //-1.00867230e-03, -7.41536000e-05, -1.09369270e-03, -1.33729200e-04};



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
	
	//gsl_matrix *A_2 = gsl_matrix_calloc(n+p-1, n+p-1);
	//if(g_flag == 0){
	//	construct_matrix(n-1, x_arr, A_2, 0);
	//}
	//else{
	//	construct_matrix(n-1, g_arr, A_2, 0);
	//}
	//int signum_2 = 0;
	//gsl_permutation *perm_2 = gsl_permutation_alloc(n+p-1);
	//gsl_linalg_LU_decomp(A_2, perm_2, &signum_2);
	//printf("A01 = %f\n", gsl_matrix_get(A,1,0));

	double max_vz, min_vz, max_vx, min_vx, max_vy, min_vy;
	double mean_change = 100000.0;
	double mean_change_best = 100000.0;
	double phi_best[n+p-1];
	
	for(int q=0; q<81; q++){
		max_vz = -100.0, min_vz = 100.0, max_vx = -100.0, min_vx = 0.0, max_vy = 0, min_vy = 100.0;
		double x, v, lambda, theta, dt, ind;
		double n_i[n];
		for(int j=0; j<n; j++){
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
		m_theta_1 = 801; 
		m_theta_2 = 201;
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
	 	double weight_for_iteration = 0.2;	
		gsl_vector *rhs_2 = gsl_vector_calloc(n+p-1);
		gsl_vector *phi_next = gsl_vector_calloc(n+p-1);
		if(g_flag == 0){
			make_vector_2(n-1, rhs_2, phi_arr, density_values_arr, x_arr);
		}
		else{
		 	make_vector_2(n-1, rhs_2, phi_arr, density_values_arr, g_arr);
		}
		
		//double phi_arr_best[n+p-1];
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
		}
		//sleep(10);
		//mean_change = mean_change/(n+p-1);
		//if(mean_change_best > mean_change){
		///	mean_change_best = mean_change;
		//	for(int l=0; l<n+p-1; l++){
		//		phi_best[l] = phi_arr[l];
		//	}
		//}	 
		if(phi_arr[0] > phi_arr[1]){
			break;
		}
		gsl_vector_free(density_values);
		gsl_vector_free(rhs_2);
		gsl_vector_free(phi_next);	
	
		end = omp_get_wtime();
	
		cpu_time_used = ((double) (end - start));
	
		printf("time = %f\n", cpu_time_used);
		//printf("check2 = %f\n", rhs_for_plot[0]);	
		printf("p: %d, n: %d\n", p, n);
		printf("mean_change = %f\n", mean_change);
		
		fout = fopen("Solution_for_check_PIC.csv", "w");
 		
		//for(int i=0; i<n+p-1; i++){
		//	printf("%f\n", rhs_for_plot[i]);
		//}
	
		for(int i=0; i<n_for_plot; i++){
			printf("%5.10f\n", n_i[i]);
			fprintf(fout, "%5.15f\n", n_i[i]);
		}
		
		double sum_value=0;
		
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
	
		
	//fclose(fout);
	//FILE *fout_3;
	FILE *fout_2;
	fout_2 = fopen("distribution_function_alpha_50_deg.txt", "w");
	//fout_3 = fopen("phi_best_alph_20_deg.txt", "w");
	//for(int l = 0; l<n+p-1; l++){
	//	fprintf(fout_3, "%f\n", phi_best[l]);
	//}
	//fclose(fout_3);
	min_vz = -5.0;
	max_vz = 5.0;
	min_vy = -5.0;
	max_vy = 5.0;
	min_vx = -5.0;
	max_vx = 0.0;
	
	gsl_matrix_free(A);
	gsl_permutation_free(perm);

	int m_vz = 200;
	int m_vx = 300;
	int m_vy = 200;
	double vz_array[m_vz], vx_array[m_vx], vy_array[m_vy];
	//double distribution_function[m_vz][m_vx][m_vy];

	linspace(min_vz, max_vz, m_vz, vz_array);
	linspace(min_vx, max_vx, m_vx, vx_array);
	linspace(min_vy, max_vy, m_vy, vy_array);
//calculate_f(double x0, double v_par0, double vx_prime0, double vy0, double t0, double dt, double *x_arr, double *phi_arr)
	double f_value;
	for(int i = 0; i<m_vz; i++){
		for(int j = 0; j<m_vx; j++){
			for(int k = 0; k<m_vy; k++){
				//if(g_flag == 0){
				//	distribution_function[i][j][k] = calculate_f(0, vz_array[i], vx_array[j], vy_array[k], 0, -0.01, x_arr, phi_arr, L);
				//}
				//else{
				//	distribution_function[i][j][k] = calculate_f(0, vz_array[i], vx_array[j], vy_array[k], 0, -0.01, g_arr, phi_arr, L);
				//}
				f_value = calculate_f(0, vz_array[i], vx_array[j], vy_array[k], 0, -0.01, x_arr, phi_arr, L);
				fprintf(fout_2, "%f %f %f %1.10f\n", vz_array[i], vx_array[j], vy_array[k], f_value);
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
