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

#include <time.h>

#include "boost/program_options.hpp"
#include "boost/filesystem/operations.hpp"

using namespace std;

namespace po = boost::program_options;
namespace bfs = boost::filesystem;

int main( int argc, char* argv[] )
{
  std::vector<std::string> parameters;
  std::vector<std::string> help;
  std::vector<float>  parametersF;
  help.push_back(" Piecewise smooth regional levelset: Regional-scalable-fitting");
  help.push_back(" Algorithm by Chunming Li 2008 IEEE Transactions on Image Processing");
  help.push_back(" Author: Hui Tang");

  po::options_description desc1("Images");
  desc1.add_options()
    ("help,h", "produce help message")
    ("initialImage,i", po::value<std::string>(), "inital binary image, .mhd file ")
    ("originalImage,o", po::value<std::string>(), "original image,.mhd file")
    ("outputImage,O",po::value<std::string>(),"output signed distance map image, .mhd file");

 po::options_description desc2("Parameters");
   desc2.add_options()
     ("parameters,P",po::value<std::vector<std::string> >(), "parameters");

  po::positional_options_description pd;
  pd.add("parameters", -1);

  po::options_description cmddescAll("Parse");
  cmddescAll.add(desc1).add(desc2);

  po::variables_map vm;
  po::parsed_options parsed=po::command_line_parser(argc, argv).options(cmddescAll).allow_unregistered().positional(pd).run();
  //po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);

  try
    {
    po::store(parsed, vm);
    po::notify(vm);
    }
  catch (std::exception& e)
    {
    cout << "Exception while parsing parameters: " << endl;
    cout << e.what() << endl;
    return EXIT_FAILURE;
    }

  // Print help message
  if (vm.count("help") || vm.size()==0)
    {
    std::cout << "========================================================================="<< std::endl;
    std::cout << "Perform piecewise smooth Regional levelset segmentation: Regional-scalable-fitting" << std::endl ;
    std::cout << "Algorithm by Chunming Li TIP 2008"<<std::endl;
    std::cout << "Author: Hui Tang"<<std::endl;
    std::cout << "========================================================================="<< std::endl;
    std::cout << "Example:RSFTermTest.exe -i initial.mhd -o originalImage.mhd -O output.mhd -P internalWeight, externalWeight, curvatureWeight, gausianBlurScale, iterationTime, RSM"<< std::endl;
    std::cout << desc1 << "\n";
    return EXIT_FAILURE;
    }

  std::vector<std::string> values(vm["parameters"].as<std::vector<std::string> >());
  for(std::vector<std::string>::iterator iter = values.begin(); values.end() != iter; ++iter)
    {
    parameters.push_back(*iter);
    std::string a = *iter;
    std::istringstream b(a);
    float f;
    b >> f;
    parametersF.push_back(f);
    }
  if( parameters.size() < 6 )
    {
    std::cout << "Error: not enough parameters provided, check help file" << std::endl;
    return EXIT_FAILURE;
    }


  time_t rawtime;
  struct tm * timeinfo;
  time ( &rawtime );
  timeinfo = localtime ( &rawtime );
  std::cout<< "RSFTermTest started at: "<<asctime (timeinfo)<< std::endl;

  // String for in- and ouput file
  std::string initialImageN;
  std::string originalImageN;
  std::string outputImageN;

  // Get intensity file
  if( vm.count( "initialImage" ) )
    {
    initialImageN = vm["initialImage"].as<std::string>();
    std::cout << "inputImage: " << initialImageN<< std::endl;
    }
  else
    {
    std::cout << "Error: no initial image provided" << std::endl;
    return EXIT_FAILURE;
    }

  const unsigned int Dimension = 3;
  typedef float                                    InputPixelType;
  typedef itk::Image< InputPixelType, Dimension >  InputImageType;

  typedef itk::ImageFileReader< InputImageType >            ReaderType;
  ReaderType::Pointer initialReader = ReaderType::New();
  initialReader->SetFileName( initialImageN );
  initialReader->Update();
  InputImageType::Pointer initial = initialReader->GetOutput();
  std::vector<float> imgExt(3, 0);
  imgExt[0] = initial->GetBufferedRegion().GetSize()[0];
  imgExt[1] = initial->GetBufferedRegion().GetSize()[1];
  imgExt[2] = initial->GetBufferedRegion().GetSize()[2];
  std::cout << "initial Image Extent " << imgExt[0] << ", " << imgExt[1] << ", " << imgExt[2] << std::endl;
  // Get intensity file

  // Get intensity file
  if (vm.count("originalImage"))
    {
    originalImageN = vm["originalImage"].as<std::string>();
    std::cout << "originalImage: " << originalImageN<< std::endl;
    }
  else
    {
    std::cout << "Error: no original image provided" << std::endl;
    return EXIT_FAILURE;
    }

  ReaderType::Pointer originalReader = ReaderType::New();
  originalReader->SetFileName( originalImageN);
  originalReader->Update();

  InputImageType::Pointer original = originalReader->GetOutput();
  std::vector<float> imgExt2(3, 0);
  imgExt2[0] = original->GetBufferedRegion().GetSize()[0];
  imgExt2[1] = original->GetBufferedRegion().GetSize()[1];
  imgExt2[2] = original->GetBufferedRegion().GetSize()[2];
  std::cout << "original image Extent " << imgExt2[0] << ", " << imgExt2[1] << ", " << imgExt2[2] << std::endl;
  if (imgExt[0]!=imgExt2[0]||imgExt[1]!=imgExt2[1]||imgExt[2]!=imgExt2[2])
    {
    std::cout << "image size should be the same!" << std::endl;
    return EXIT_FAILURE;
    }

  if (vm.count("outputImage"))
    {
    outputImageN = vm["outputImage"].as<std::string>();
    std::cout << "outputImage: " << outputImageN<< std::endl;
    }
  else
    {
    std::cout << "Error: no output image name provided" << std::endl;
    return EXIT_FAILURE;
    }

  typedef itk::WhitakerSparseLevelSetImage < InputPixelType, Dimension > SparseLevelSetType;
  //typedef itk::LevelSetDenseImageBase< InputImageType > SparseLevelSetType;
  typedef itk::BinaryImageToLevelSetImageAdaptor< InputImageType,
      SparseLevelSetType> BinaryToSparseAdaptorType;

  BinaryToSparseAdaptorType::Pointer adaptor = BinaryToSparseAdaptorType::New();
  adaptor->SetInputImage( initial );
  adaptor->Initialize();

  // Here get the resulting level-set function

  typedef SparseLevelSetType::Pointer SparseLevelSetTypePointer;
  SparseLevelSetTypePointer level_set = adaptor->GetLevelSet();

  // Create here the bounds in which this level-set can evolved.

  // There is only one level-set, so we fill 1 list with only one element which
  // correspondongs to the level-set identifier.
  typedef itk::IdentifierType         IdentifierType;
  typedef std::list< IdentifierType > IdListType;

  IdListType list_ids;
  list_ids.push_back( 1 );

  // We create one image where for each pixel we provide which level-set exists.
  // In this example the first level-set is defined on the whole image.
  typedef itk::Image< IdListType, Dimension >               IdListImageType;
  IdListImageType::Pointer id_image = IdListImageType::New();
  id_image->SetRegions( initial->GetLargestPossibleRegion() );
  id_image->Allocate();
  id_image->FillBuffer( list_ids );

  typedef itk::Image< short, Dimension >                     CacheImageType;
  typedef itk::LevelSetDomainMapImageFilter< IdListImageType, CacheImageType >
      DomainMapImageFilterType;
  DomainMapImageFilterType::Pointer domainMapFilter = DomainMapImageFilterType::New();
  domainMapFilter->SetInput( id_image );
  domainMapFilter->Update();

  // Define the Heaviside function
  typedef SparseLevelSetType::OutputRealType LevelSetOutputRealType;

  typedef itk::AtanRegularizedHeavisideStepFunction< LevelSetOutputRealType,
  LevelSetOutputRealType > HeavisideFunctionBaseType;
  HeavisideFunctionBaseType::Pointer heaviside = HeavisideFunctionBaseType::New();
  heaviside->SetEpsilon( 1.5 );

  // Insert the levelsets in a levelset container
  typedef itk::LevelSetContainer< IdentifierType, SparseLevelSetType >
    LevelSetContainerType;
  typedef itk::LevelSetEquationTermContainer< InputImageType, LevelSetContainerType >
    TermContainerType;

  LevelSetContainerType::Pointer lscontainer = LevelSetContainerType::New();
  lscontainer->SetHeaviside( heaviside );
  lscontainer->SetDomainMapFilter( domainMapFilter );

  lscontainer->AddLevelSet( 0, level_set );

  std::cout << "Level set container created" << std::endl;

  typedef itk::LevelSetEquationSparseRSFTerm< InputImageType,
      LevelSetContainerType > RSFTermType;

  RSFTermType::Pointer cvTerm0 = RSFTermType::New();
  cvTerm0->SetInput( original  );
  cvTerm0->SetInternalCoefficient( parametersF[0]  );
  std::cout<<"InternalCoefficient:"<<parametersF[0]<<std::endl;
  cvTerm0->SetExternalCoefficient(  parametersF[1] );
  std::cout<<"ExternalCoefficient:"<<parametersF[1]<<std::endl;
  cvTerm0->SetGaussianBlurScale(parametersF[3] );
  std::cout<<"GaussianBlurScale:"<<parametersF[3]<<std::endl;
  cvTerm0->SetCurrentLevelSetId( 0 );
  cvTerm0->SetLevelSetContainer( lscontainer );

  typedef itk::LevelSetEquationCurvatureTerm< InputImageType,
      LevelSetContainerType > CurvatureTermType;

  CurvatureTermType::Pointer curvatureTerm0 =  CurvatureTermType::New();
  curvatureTerm0->SetCoefficient( parametersF[2] );
  std::cout<<"GeodesicCurvatureWeight:"<<parametersF[2]<<std::endl;
  curvatureTerm0->SetCurrentLevelSetId( 0 );
  curvatureTerm0->SetLevelSetContainer( lscontainer );

  // **************** CREATE ALL EQUATIONS ****************

  // Create Term Container which corresponds to the combination of terms in the PDE.

  TermContainerType::Pointer termContainer0 = TermContainerType::New();

  termContainer0->SetInput( original  );
  termContainer0->SetCurrentLevelSetId( 0 );
  termContainer0->SetLevelSetContainer( lscontainer );
  termContainer0->AddTerm( 0, cvTerm0  );
  termContainer0->AddTerm( 1, curvatureTerm0 );


  typedef itk::LevelSetEquationContainer< TermContainerType > EquationContainerType;
  EquationContainerType::Pointer equationContainer = EquationContainerType::New();
  equationContainer->AddEquation( 0, termContainer0 );
  equationContainer->SetLevelSetContainer( lscontainer );

  typedef itk::LevelSetEvolutionNumberOfIterationsStoppingCriterion< LevelSetContainerType >
    StoppingCriterionType;
  StoppingCriterionType::Pointer criterion = StoppingCriterionType::New();
  criterion->SetNumberOfIterations(  parametersF[4] );
  std::cout<<"NumberOfIterations:"<<parametersF[4]<<endl;

  if( criterion->GetNumberOfIterations() != parametersF[4] )
    {
    return EXIT_FAILURE;
    }

  criterion->SetRMSChangeAccumulator( parametersF[5]);
  std::cout<<"RMSChangeAccumulator:"<<parametersF[5]<<endl;

  if( criterion->GetRMSChangeAccumulator() != parametersF[5] )
    {
    return EXIT_FAILURE;
    }

  typedef itk::LevelSetEvolution< EquationContainerType, SparseLevelSetType > LevelSetEvolutionType;
  LevelSetEvolutionType::Pointer evolution = LevelSetEvolutionType::New();

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

  InputImageType::Pointer outputImage = InputImageType::New();
  outputImage->SetRegions( original->GetLargestPossibleRegion() );
  outputImage->CopyInformation( original );
  outputImage->Allocate();
  outputImage->FillBuffer( 0 );

  typedef itk::ImageRegionIteratorWithIndex< InputImageType > OutputIteratorType;
  OutputIteratorType oIt( outputImage, outputImage->GetLargestPossibleRegion() );
  oIt.GoToBegin();

  InputImageType::IndexType idx;

  while( !oIt.IsAtEnd() )
    {
    idx = oIt.GetIndex();
    //oIt.Set( level_set->GetLabelMap()->GetPixel(idx) );
    oIt.Set( level_set->Evaluate(idx) );
    ++oIt;
    }

  typedef itk::ImageFileWriter< InputImageType >     OutputWriterType;
  OutputWriterType::Pointer writer = OutputWriterType::New();
  writer->SetFileName(outputImageN);
  //writer->SetInput( binary);
  writer->SetInput( outputImage );

  try
    {
    writer->Update();

    std::cout << "outputfile is saved as " << outputImageN<<std::endl;
    }
  catch ( itk::ExceptionObject& err )
    {
    std::cout << err << std::endl;
    return EXIT_FAILURE;
    }

  time_t rawtime1;
  struct tm * timeinfo1;
  time ( &rawtime1 );
  timeinfo1 = localtime ( &rawtime1 );

  return EXIT_SUCCESS;
}
