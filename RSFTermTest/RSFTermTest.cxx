/*=========================================================================
 *
 *  Copyright Biomedical Imaging Group Rotterdam, Erasmus MC, the Netherlands
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/


#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkLevelSetDomainMapImageFilter.h"
#include "itkLevelSetContainerBase.h"
#include "itkLevelSetEquationSparseRSFTerm.h"
#include "itkLevelSetEquationChanAndVeseInternalTerm.h"
#include "itkLevelSetEquationCurvatureTerm.h"
#include "itkLevelSetEquationTermContainer.h"
#include "itkLevelSetEquationContainer.h"
#include "itkLevelSetEvolution.h"
#include "itkAtanRegularizedHeavisideStepFunction.h"
#include "itkWhitakerSparseLevelSetImage.h"
#include "itkLevelSetEvolutionNumberOfIterationsStoppingCriterion.h"
#include "itkBinaryImageToLevelSetImageAdaptor.h"
#include "itkLevelSetEquationCurvatureTerm.h"
#include "itkLevelSetContainer.h"

template <unsigned int ImageDimension>
int RSFTest( int argc, char *argv[] )
{
	// String for in- and ouput file
	std::string initialImageN;
	std::string originalImageN;
	std::string outputImageN;


	typedef float                                    InputPixelType;
	typedef itk::Image< InputPixelType, ImageDimension >  InputImageType;
	typedef itk::ImageFileReader< InputImageType >         ReaderType;
	typename ReaderType::Pointer initialReader = ReaderType::New();
	initialReader->SetFileName( argv[1] );

	try
	{
	  initialReader->Update();
	}
	catch (std::exception& e)
	{
		std::cout << "Exception while reading image " <<argv[1]<<std::endl;
		std::cout << e.what() << std::endl;
		exit(-1);
	}

	typename InputImageType::Pointer initial = initialReader->GetOutput();
	typename InputImageType::SizeType imgExt;
	imgExt[0] = initial->GetBufferedRegion().GetSize()[0];
	imgExt[1] = initial->GetBufferedRegion().GetSize()[1];
	imgExt[2] = initial->GetBufferedRegion().GetSize()[2];
	
	typename ReaderType::Pointer originalReader = ReaderType::New();
	originalReader->SetFileName( argv[2]);
	try
	{
		originalReader->Update();
	}
	catch (std::exception& e)
	{
		std::cout << "Exception while reading image " << argv[2] << std::endl;
		std::cout << e.what() << std::endl;
		exit(-1);
	}

	typename InputImageType::Pointer original = originalReader->GetOutput();
	typename InputImageType::SizeType imgExt2;
	imgExt2[0] = original->GetBufferedRegion().GetSize()[0];
	imgExt2[1] = original->GetBufferedRegion().GetSize()[1];
	imgExt2[2] = original->GetBufferedRegion().GetSize()[2];
	for (int i=0; i<ImageDimension;i++)
	{
		if( imgExt[i]!=imgExt2[i])
		{   
			std::cout << "input image size should be the same!" << std::endl;
			return EXIT_FAILURE;
		}
	}
   
	typedef itk::WhitakerSparseLevelSetImage < InputPixelType, ImageDimension > SparseLevelSetType;
	typedef itk::BinaryImageToLevelSetImageAdaptor< InputImageType,
		SparseLevelSetType> BinaryToSparseAdaptorType;

	typename BinaryToSparseAdaptorType::Pointer adaptor = BinaryToSparseAdaptorType::New();
	adaptor->SetInputImage( initial );
	adaptor->Initialize();

	typedef typename SparseLevelSetType::Pointer SparseLevelSetTypePointer;
	SparseLevelSetTypePointer levelset = adaptor->GetLevelSet();

	typedef itk::IdentifierType         IdentifierType;
	typedef std::list< IdentifierType > IdListType;

	IdListType listIds;
	listIds.push_back( 1 );

	typedef itk::Image< IdListType, ImageDimension >               IdListImageType;
	typename IdListImageType::Pointer idimage = IdListImageType::New();
	idimage->SetRegions( initial->GetLargestPossibleRegion() );
	idimage->Allocate();
	idimage->FillBuffer( listIds );

	typedef itk::Image< short, ImageDimension >                     CacheImageType;
	typedef itk::LevelSetDomainMapImageFilter< IdListImageType, CacheImageType >
		DomainMapImageFilterType;
	typename DomainMapImageFilterType::Pointer domainMapFilter = DomainMapImageFilterType::New();
	domainMapFilter->SetInput( idimage );
	domainMapFilter->Update();

	// Define the Heaviside function
	typedef typename SparseLevelSetType::OutputRealType LevelSetOutputRealType;

	typedef itk::AtanRegularizedHeavisideStepFunction< LevelSetOutputRealType,LevelSetOutputRealType > HeavisideFunctionType;
	typename HeavisideFunctionType::Pointer heaviside = HeavisideFunctionType::New();
	heaviside->SetEpsilon( 1.5 );

	// Insert the levelsets in a levelset container
	typedef itk::LevelSetContainer< IdentifierType, SparseLevelSetType >
		LevelSetContainerType;
	typedef itk::LevelSetEquationTermContainer< InputImageType, LevelSetContainerType >
		TermContainerType;

	typename LevelSetContainerType::Pointer lsContainer = LevelSetContainerType::New();
	lsContainer->SetHeaviside( heaviside );
	lsContainer->SetDomainMapFilter( domainMapFilter );

	lsContainer->AddLevelSet( 0, levelset );
    std::cout << std::endl;
	std::cout << "Level set container created" << std::endl;

	typedef itk::LevelSetEquationSparseRSFTerm< InputImageType,
		LevelSetContainerType > RSFTermType;

	typename RSFTermType::Pointer rsfTerm0 = RSFTermType::New();
	rsfTerm0->SetInput( original  );
	rsfTerm0->SetInternalCoefficient( atof(argv[4])  );
	std::cout << "InternalCoefficient:" << atof(argv[4])<< std::endl;
	rsfTerm0->SetExternalCoefficient(  atof(argv[5]) );
	std::cout << "ExternalCoefficient:" << atof(argv[5])<< std::endl;
	rsfTerm0->SetGaussianBlurScale(atof(argv[7]) );
	std::cout << "GaussianBlurScale:" << atof(argv[7])<< std::endl;
	rsfTerm0->SetCurrentLevelSetId( 0 );
	rsfTerm0->SetLevelSetContainer( lsContainer );

	typedef itk::LevelSetEquationCurvatureTerm< InputImageType,
		LevelSetContainerType > CurvatureTermType;

	typename CurvatureTermType::Pointer curvatureTerm0 =  CurvatureTermType::New();
	curvatureTerm0->SetCoefficient( atof(argv[6]) );
	std::cout << "CurvatureWeight:" << atof(argv[6]) << std::endl;
	curvatureTerm0->SetCurrentLevelSetId( 0 );
	curvatureTerm0->SetLevelSetContainer( lsContainer );

	// **************** CREATE ALL EQUATIONS ****************

	// Create Term Container which corresponds to the combination of terms in the PDE.

	typename TermContainerType::Pointer termContainer0 = TermContainerType::New();

	termContainer0->SetInput( original  );
	termContainer0->SetCurrentLevelSetId( 0 );
	termContainer0->SetLevelSetContainer( lsContainer );
	termContainer0->AddTerm( 0, rsfTerm0  );
	termContainer0->AddTerm( 1, curvatureTerm0 );


	typedef itk::LevelSetEquationContainer< TermContainerType > EquationContainerType;
	typename EquationContainerType::Pointer equationContainer = EquationContainerType::New();
	equationContainer->AddEquation( 0, termContainer0 );
	equationContainer->SetLevelSetContainer( lsContainer );

	typedef itk::LevelSetEvolutionNumberOfIterationsStoppingCriterion< LevelSetContainerType >
		StoppingCriterionType;
	typename StoppingCriterionType::Pointer criterion = StoppingCriterionType::New();
	criterion->SetNumberOfIterations(  atof(argv[8]) );
	std::cout << "NumberOfIterations:" << atof(argv[8]) << std::endl;

	if( criterion->GetNumberOfIterations() != atof(argv[8]))
	{
		return EXIT_FAILURE;
	}

	criterion->SetRMSChangeAccumulator( 0.001);

	if( criterion->GetRMSChangeAccumulator() != 0.001 )
	{
		return EXIT_FAILURE;
	}

	typedef itk::LevelSetEvolution< EquationContainerType, SparseLevelSetType > LevelSetEvolutionType;
	typename LevelSetEvolutionType::Pointer evolution = LevelSetEvolutionType::New();

	evolution->SetEquationContainer( equationContainer );
	evolution->SetStoppingCriterion( criterion );
	evolution->SetLevelSetContainer( lsContainer );

	try
	{
		evolution->Update();
	}
	catch ( itk::ExceptionObject& err )
	{
		std::cerr << err << std::endl;
		std::cout << "ERROR::can not update evolution!!" << std::endl;

		return EXIT_FAILURE;
	}

	typename InputImageType::Pointer outputImage = InputImageType::New();
	outputImage->SetRegions( original->GetLargestPossibleRegion() );
	outputImage->CopyInformation( original );
	outputImage->Allocate();
	outputImage->FillBuffer( 0 );

	typedef itk::ImageRegionIteratorWithIndex< InputImageType > OutputIteratorType;
	OutputIteratorType oIt( outputImage, outputImage->GetLargestPossibleRegion() );
	oIt.GoToBegin();

	typename InputImageType::IndexType idx;

	while( !oIt.IsAtEnd() )
	{
		idx = oIt.GetIndex();
		oIt.Set( levelset->Evaluate(idx) );
		++oIt;
	}

	typedef itk::ImageFileWriter< InputImageType >     OutputWriterType;
	typename OutputWriterType::Pointer writer = OutputWriterType::New();
	writer->SetFileName(argv[3]);
	writer->SetInput( outputImage );

	try
	{
		writer->Update();

		std::cout << "outputfile is saved as " << argv[3]<<std::endl;
	}
	catch ( itk::ExceptionObject& err )
	{
		std::cout << err << std::endl;
		return EXIT_FAILURE;
	}


	return EXIT_SUCCESS;
}

int main( int argc, char* argv[] )
{ 

    std::cout << "================================================================================" <<std::endl;
	std::cout << "A test for the itk class itk::LevelSetEqationSparseRSFTerm" <<std::endl<<std::endl;
	std::cout << "Which is an implementation of the paper published by: "<<std::endl<<std::endl;
	std::cout << "C. Li, C. Kao, J. C. Gore, and Z. Ding. "<<std::endl<<std::endl;
	std::cout << "Minimization of region-scalable fitting energy for image segmentation. IEEE Trans Image Processing, 17 (10):1940--1949, 2008." <<std::endl<<std::endl;
	std::cout << "=================================================================================" <<std::endl;
	if ( argc < 10 )
	{
		std::cerr << "Usage: " << argv[0] << " [initial image (binary image)] [original image] [output image] [internal coefficient] [external coefficient] [curvature coefficient] [gaussian scale] [iteration time] [image dimension]" << std::endl;
		return( EXIT_FAILURE );
	}
	switch( atoi( argv[9] ) )
	{
	case 2:
		RSFTest<2>( argc, argv );
		break;
	case 3:
		RSFTest<3>( argc, argv );
		break;
	default:
		std::cerr << "Unsupported dimension" << std::endl;
		return( EXIT_FAILURE );
	} 
}

