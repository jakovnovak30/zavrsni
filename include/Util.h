/**
 * @file
 * @brief definicija korisnih globalnih funkcija i konstanti
 * @author Jakov Novak
 */

#pragma once

#include<CL/cl.h>

/**
 * @brief globalni OpenCL kontekst koje se inicijalizira prilikom pokretanja initCL funkcije
 */
extern cl_context globalContext;
/**
 * @brief globalni uređaj koje se inicijalizira prilikom pokretanja initCL funkcije
 */
extern cl_device_id globalDevice;
/**
 * @brief globalni OpenCL komandni red koji se inicijalizira prilikom pokretanja initCL funkcije
 */
extern cl_command_queue globalQueue;

/**
 * @brief funkcija koja na temelju OpenCL koda baca iznimke ili ne vraća ništa
 *
 * @param value 
 * @throws std::runtime_error baca se u slučaju da je value != 0 te se ispisuje razlog pogreške OpenCL-a
 */
void checkError(int value);

/**
 * @brief funkcija koja na temelju pokazivača na program i jezgru te dodatnih parametara prevodi određeni OpenCL kod te ga sprema u parametre
 *
 * Ideja funkcije je da se program i kernel inicijaliziraju na NULL te da se za vrijeme izvođenje provjeri jesu li oni NULL i onda se prevode na temelju zadanog izvornog koda ili funkcija ništa ne radi. Na taj način se štede resursi prilikom konstrukcije nekih objekata jer postoji mogućnost da se njihove jezgre nikad neće koristiti.
 *
 * @param program pokazivač na program čija vrijednost može biti NULL (treba ga prevesti) ili nešto drugo (prevođen je)
 * @param kernel pokazivač na jezgru čija vrijednost može biti NULL (treba ju postaviti) ili nešto drugo (funkcija ne radi ništa)
 * @param kernel_name string koji određuje ime jezgre unutar OpenCL izvornog koda
 * @param srcStr OpenCL izvorni kod
 * @param srcLen duljina OpenCL izvornog koda
 */
void buildIfNeeded(cl_program *program, cl_kernel *kernel, const char *kernel_name,
                   const char **srcStr, const size_t *srcLen);

/**
 * @brief funkcija koda na temelju uređaja i konteksta postavlja globalne konstante
 *
 * @param device uređaj za kojeg želimo postaviti da je globalan
 * @param context kontekst koje će biti globalan
 */
void initCL(cl_device_id device, cl_context context);
/**
 * @brief funkcija koja traži grafičku karticu tvrtke NVIDIA te zatim delegira postavljanje konteksta funkciji initCL
 */
void initCL_nvidia();
/**
 * @brief funkcija koja oslobađa sve OpenCL resurse zajedno s kontekstom i uređajem
 */
void freeCL();
