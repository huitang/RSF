/*=========================================================================
 *
 *  Copyright Insight Software Consortium
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
#define BOOST_EXCEPTION_DISABLE 1

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkLevelSetDomainMapImageFilter.h"
#include "itkLevelSetContainerBase.h"
#include "itkLevelSetEquationSparseRSFTerm.h"
#include "itkLevelSetEquationCurvatureTerm.h"
#include "itkLevelSetEquationTermContainer.h"
#include "itkLevelSetEquationContainer.h"
#include "itkAtanRegularizedHeavisideStepFunction.h"
#include "itkLevelSetEvolution.h"
#include "itkWhitakerSparseLevelSetImage.h"
#include "itkLevelSetEvolutionNumberOfIterationsStoppingCriterion.h"
#include "itkBinaryImageToLevelSetImageAdaptor.h"
#include "itkGradientMagnitudeRecursiveGaussianImageFilter.h"
#include "itkLevelSetEquationCurvatureTerm.h"
#include "itkLevelSetEquationAdvectionTerm.h"
#include "itkLevelSetContainer.h"
#include "itkLevelSetEquationLaplacianTerm.h"
#include "itkAtanRegularizedHeavisideStepFunction.h"
#include "itkBinaryImageToLevelSetImageAdaptor.h"
#include "itkLevelSetEvolutionNumberOfIterationsStoppingCriterion.h"
#include "itkScalarChanAndVeseSparseLevelSetImageFilter.h"
#include "itkLevelSetEquationChanAndVeseInternalTerm.h"

using namespace std;

template <unsigned int ImageDimension>
int RSFTest( int argc, char *argv[] )
{
	// String for in- and ouput file
	std::string initialImageN;
	std::string originalImageN;
	std::string outputImageN;


	typedef float                                    InputPixelType;
	typedef itk::Image< InputPixelType, ImageDimension >  InputImageType;
	typedef itk::ImageFileReader< InputImageType >            ReaderType;
	typename ReaderType::Pointer initialReader = ReaderType::New();
	initialReader->SetFileName( argv[1] );
	initialReader->Update();
	typename InputImageType::Pointer initial = initialReader->GetOutput();
	std::vector<float> imgExt(3, 0);
	imgExt[0] = initial->GetBufferedRegion().GetSize()[0];
	imgExt[1] = initial->GetBufferedRegion().GetSize()[1];
	imgExt[2] = initial->GetBufferedRegion().GetSize()[2];
	std::cout << "initial Image Extent " << imgExt[0] << ", " << imgExt[1] << ", " ;
		if (ImageDimension==3)
		{
			std::cout<< imgExt[2] << std::endl;
		}

	typename ReaderType::Pointer originalReader = ReaderType::New();
	originalReader->SetFileName( argv[2]);
	originalReader->Update();

	typename InputImageType::Pointer original = originalReader->GetOutput();
	std::vector<float> imgExt2(3, 0);
	imgExt2[0] = original->GetBufferedRegion().GetSize()[0];
	imgExt2[1] = original->GetBufferedRegion().GetSize()[1];
	imgExt2[2] = original->GetBufferedRegion().GetSize()[2];
	std::cout<< std::endl;
	std::cout << "original image Extent " << imgExt2[0] << ", " << imgExt2[1] << ", "; 
	if (ImageDimension==3)
	{
		std::cout<< imgExt2[2] << std::endl;
	}
	if (ImageDimension==3){
		if(imgExt[0]!=imgExt2[0]||imgExt[1]!=imgExt2[1]||imgExt[2]!=imgExt2[2])
	   {
		std::cout <<"3D image size should be the same!" << std::endl;
		return EXIT_FAILURE;
	   }
	}
	if (ImageDimension==2) {
		if( imgExt[0]!=imgExt2[0]||imgExt[1]!=imgExt2[1])
	   {
		std::cout <<"2D image image size should be the same!" << std::endl;
		return EXIT_FAILURE;
	   }
	}
   
	typedef itk::WhitakerSparseLevelSetImage < InputPixelType, ImageDimension > SparseLevelSetType;
	//typedef itk::LevelSetDenseImageBase< InputImageType > SparseLevelSetType;
	typedef itk::BinaryImageToLevelSetImageAdaptor< InputImageType,
		SparseLevelSetType> BinaryToSparseAdaptorType;

	typename BinaryToSparseAdaptorType::Pointer adaptor = BinaryToSparseAdaptorType::New();
	adaptor->SetInputImage( initial );
	adaptor->Initialize();

	typedef typename SparseLevelSetType::Pointer SparseLevelSetTypePointer;
	SparseLevelSetTypePointer level_set = adaptor->GetLevelSet();

	typedef itk::IdentifierType         IdentifierType;
	typedef std::list< IdentifierType > IdListType;

	IdListType list_ids;
	list_ids.push_back( 1 );

	typedef itk::Image< IdListType, ImageDimension >               IdListImageType;
	typename IdListImageType::Pointer id_image = IdListImageType::New();
	id_image->SetRegions( initial->GetLargestPossibleRegion() );
	id_image->Allocate();
	id_image->FillBuffer( list_ids );

	typedef itk::Image< short, ImageDimension >                     CacheImageType;
	typedef itk::LevelSetDomainMapImageFilter< IdListImageType, CacheImageType >
		DomainMapImageFilterType;
	typename DomainMapImageFilterType::Pointer domainMapFilter = DomainMapImageFilterType::New();
	domainMapFilter->SetInput( id_image );
	domainMapFilter->Update();

	// Define the Heaviside function
	typedef typename SparseLevelSetType::OutputRealType LevelSetOutputRealType;

	typedef itk::AtanRegularizedHeavisideStepFunction< LevelSetOutputRealType,
		LevelSetOutputRealType > HeavisideFunctionBaseType;
	typename HeavisideFunctionBaseType::Pointer heaviside = HeavisideFunctionBaseType::New();
	heaviside->SetEpsilon( 1.5 );

	// Insert the levelsets in a levelset container
	typedef itk::LevelSetContainer< IdentifierType, SparseLevelSetType >
		LevelSetContainerType;
	typedef itk::LevelSetEquationTermContainer< InputImageType, LevelSetContainerType >
		TermContainerType;

	typename LevelSetContainerType::Pointer lscontainer = LevelSetContainerType::New();
	lscontainer->SetHeaviside( heaviside );
	lscontainer->SetDomainMapFilter( domainMapFilter );

	lscontainer->AddLevelSet( 0, level_set );
    std::cout<< std::endl;
	std::cout << "Level set container created" << std::endl;

	typedef itk::LevelSetEquationSparseRSFTerm< InputImageType,
		LevelSetContainerType > RSFTermType;

	typename RSFTermType::Pointer cvTerm0 = RSFTermType::New();
	cvTerm0->SetInput( original  );
	cvTerm0->SetInternalCoefficient( atoi(argv[4])  );
	std::cout<<"InternalCoefficient:"<<atoi(argv[4])<<std::endl;
	cvTerm0->SetExternalCoefficient(  atoi(argv[5]) );
	std::cout<<"ExternalCoefficient:"<<atoi(argv[5])<<std::endl;
	cvTerm0->SetGaussianBlurScale(atoi(argv[7]) );
	std::cout<<"GaussianBlurScale:"<<atoi(argv[7])<<std::endl;
	cvTerm0->SetCurrentLevelSetId( 0 );
	cvTerm0->SetLevelSetContainer( lscontainer );

	typedef itk::LevelSetEquationCurvatureTerm< InputImageType,
		LevelSetContainerType > CurvatureTermType;

	typename CurvatureTermType::Pointer curvatureTerm0 =  CurvatureTermType::New();
	curvatureTerm0->SetCoefficient( atoi(argv[6]) );
	std::cout<<"GeodesicCurvatureWeight:"<<atoi(argv[6])<<std::endl;
	curvatureTerm0->SetCurrentLevelSetId( 0 );
	curvatureTerm0->SetLevelSetContainer( lscontainer );

	// **************** CREATE ALL EQUATIONS ****************

	// Create Term Container which corresponds to the combination of terms in the PDE.

	typename TermContainerType::Pointer termContainer0 = TermContainerType::New();

	termContainer0->SetInput( original  );
	termContainer0->SetCurrentLevelSetId( 0 );
	termContainer0->SetLevelSetContainer( lscontainer );
	termContainer0->AddTerm( 0, cvTerm0  );
	termContainer0->AddTerm( 1, curvatureTerm0 );


	typedef itk::LevelSetEquationContainer< TermContainerType > EquationContainerType;
	typename EquationContainerType::Pointer equationContainer = EquationContainerType::New();
	equationContainer->AddEquation( 0, termContainer0 );
	equationContainer->SetLevelSetContainer( lscontainer );

	typedef itk::LevelSetEvolutionNumberOfIterationsStoppingCriterion< LevelSetContainerType >
		StoppingCriterionType;
	typename StoppingCriterionType::Pointer criterion = StoppingCriterionType::New();
	criterion->SetNumberOfIterations(  atoi(argv[8]) );
	std::cout<<"NumberOfIterations:"<<atoi(argv[8])<<endl;

	if( criterion->GetNumberOfIterations() != atoi(argv[8]))
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
	evolution->SetLevelSetContainer( lscontainer );

	try
	{
		evolution->Update();
	}
	catch ( itk::ExceptionObject& err )
	{
		std::cerr << err << std::endl;
		std::cout<<"ERROR::can not update evolution!!"<<std::endl;

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
		//oIt.Set( level_set->GetLabelMap()->GetPixel(idx) );
		oIt.Set( level_set->Evaluate(idx) );
		++oIt;
	}

	typedef itk::ImageFileWriter< InputImageType >     OutputWriterType;
	typename OutputWriterType::Pointer writer = OutputWriterType::New();
	writer->SetFileName(argv[3]);
	//writer->SetInput( binary);
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

	if ( argc < 10 )
	{
		std::cerr << "Usage: " << argv[0] << " [initial image (binary image)] [original image] [output image] [internal weight] [external weight] [curvature weight] [gaussian scale] [iteration time] [image dimension]" << std::endl;
		exit( EXIT_FAILURE );
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
		exit( EXIT_FAILURE );
	} 
}

