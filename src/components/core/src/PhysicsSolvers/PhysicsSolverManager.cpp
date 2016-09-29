/*
 * PhysicsSolverManager.cpp
 *
 *  Created on: Sep 7, 2016
 *      Author: rrsettgast
 */

#include "PhysicsSolverManager.hpp"

#include "../../../cxx-utilities/src/src/DocumentationNode.hpp"
#include "SolverBase.hpp"
#include "pugixml/src/pugixml.hpp"

namespace geosx
{

using namespace dataRepository;
using namespace cxx_utilities;

PhysicsSolverManager::PhysicsSolverManager( std::string const & name,
                                            ManagedGroup * const parent ):
  ManagedGroup( name, parent)
{
}

PhysicsSolverManager::~PhysicsSolverManager()
{
}



SolverBase & PhysicsSolverManager::CreateSolver( string const & solverCatalogKey, string const & solverName )
{
  std::unique_ptr<SolverBase> solver = SolverBase::CatalogInterface::Factory( solverCatalogKey, solverName, this );
  SolverBase & rval = this->RegisterGroup( solverName, std::move(solver) );

  return rval;
}

void PhysicsSolverManager::ReadXML( dataRepository::ManagedGroup& domain,
                                    pugi::xml_node const & problemNode )
{
  // Set the appropriate schema type
  cxx_utilities::DocumentationNode * const docNode = this->getDocumentationNode();
  docNode->setSchemaType("UniqueNode");

  // Store a list of available solvers
  RegisterViewWrapper<string_array>(keys::solverNames);

  // Solvers
  pugi::xml_node topLevelNode = problemNode.child("Solvers");
  std::cout << "Solvers:" << std::endl;
  if (topLevelNode == NULL)
  {
    throw std::invalid_argument("Solver block not present in input xml file!");
  }
  else
  {
    // The push_back and insert methods don't seem to work, so manually resize the manager to hold the solver list
    long nSolvers = std::distance(topLevelNode.children().begin(), topLevelNode.children().end());
    this->resize(nSolvers);
    ViewWrapper<string_array>::rtype  solverNames = getData<string_array>(keys::solverNames);
    int solverNumber = 0;

    for (pugi::xml_node solverNode=topLevelNode.first_child(); solverNode; solverNode=solverNode.next_sibling())
    {
      std::cout << "   " << solverNode.name() << std::endl;

      // Register the new solver
      std::string solverID = solverNode.attribute("name").value();
      SolverBase & newSolver = CreateSolver( solverNode.name(), solverID );

      // Set the documentation node
      newSolver.SetDocumentationNodes();

      // Register fields in the solver and parse options
      newSolver.Registration( &domain );
      newSolver.ReadXML(solverNode );
      solverNames[solverNumber] = solverID;
      solverNumber++;
    }
  }

}


} /* namespace geosx */
